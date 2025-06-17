import torch
import os
import gc
import time
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from early_stopping import EarlyStopping
from utils import set_random_seed, get_n_params, select_causal_metapaths, train_search, write_file,train
from params import parse_args_search, parse_args_study
from evaluate_util import score, evaluate, full_evaluate, self_f1_score
from data_prepare import data_prepare, index_prepare, get_target_info, add_eval_loader
from data_process import adjs_process, neighbour_aggregation, label_feat_propagate
from mdoel_search import CMS_Se
from model_study import CMS

def run_search(args):
    device = args.device
    global best_meta, best_label

    g, adjs, init_labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full = data_prepare(args)

    # 这句代码的整体作用是将邻接矩阵中每条边的权重重新设置为该边起始节点度的倒数。这样做的目的是进行归一化处理，使得每个节点连接的边权重之和为1，通常用于消息传递中的归一化步骤。
    adjs = adjs_process(adjs)

    # 对节点的索引重新赋值（由于可能去掉了一些没有标签的节点，所以需要对节点的索引进行重新赋值）
    labels, init2sort, sort2init, train_node_nums, total_num_nodes, train_val_point, val_test_point = index_prepare(train_nid,val_nid,test_nid,args.dataset,dl,init_labels)

    # 根据数据集的不同，设置目标节点类型tgt_type、所有节点类型node_types和额外元路径extra_metapath。
    tgt_type, extra_metapath = get_target_info(args.dataset, args.num_hops)

    # 打印当前特征传播的跳数args.num_hops。
    print(f'Current num hops = {args.num_hops}')

    # 平均聚合，进行k跳特征传播
    feats, data_size = neighbour_aggregation(args, g, extra_metapath, tgt_type, init2sort)

    # 设置检查点文件夹
    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)

    # 如果 args.amp 为真，则初始化 GradScaler，用于混合精度训练。
    if args.amp:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    # 对于非 IMDB 数据集，将标签转换为 long 类型，并移动到选定设备。
    # 对于 IMDB 数据集，将标签转换为 float 类型，并移动到选定设备。
    if args.dataset != 'IMDB':
        labels_cuda = labels.long().to(args.device)
    else:
        labels = labels.float()
        labels_cuda = labels.to(args.device)

    # 目标节点数量
    num_target_nodes = dl.nodes['count'][0]

    # 开始训练
    for seed in args.seeds:
        set_random_seed(seed)
        epochs = args.num_epochs

        stopper = EarlyStopping(checkpt_folder,patience=args.patience)

        # 标签特征传播
        label_feats = label_feat_propagate(args, num_target_nodes,tgt_type,num_classes,init_labels,train_nid,val_nid,test_nid,init2sort,adjs)

        # 创建训练数据加载器 train_loader，随机打乱训练节点索引，并设定 batch_size 和 drop_last 参数。
        # todo 待测试，目前是验证集占0.2，不确定要不要调到验证集占0.5
        train_loader = torch.utils.data.DataLoader(
            torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)
        # 多了一个val_loader，只有验证集数据
        val_loader = torch.utils.data.DataLoader(torch.arange(train_val_point, val_test_point), batch_size=args.batch_size, shuffle=True, drop_last=False)

        # 评估数据加载器 eval_loader 和多余数据加载器 full_loader。
        eval_loader, full_loader = add_eval_loader(args.batch_size, total_num_nodes, num_target_nodes, feats, label_feats)


        # 检查计算出的 args.ns 是否超过了特征和标签特征的总数量。如果超过，则将 args.ns 设置为特征和标签特征总数。这确保了采样数量不会超过可用特征的数量。
        # todo 待优化，后续可以根据元路径的因果效应来决定需要多少条元路径（比如说如果这条元路径的因果效应为正，则就将其加进去）
        if args.ns > (len(feats) + len(label_feats)):
            args.ns = (len(feats) + len(label_feats))


        # 释放显存和手动触发垃圾回收。
        torch.cuda.empty_cache()
        gc.collect()

        model = CMS_Se(args.hidden_size, num_classes, feats.keys(), label_feats.keys(), tgt_type,
                         args.dropout, args.input_drop, device, args.num_final, args.residual, bns=args.bns,
                         data_size=data_size, num_sampled=args.ns)

        model = model.to(args.device)

        # 打印模型参数总数（在 args.seed 为初始值时）。
        if seed == args.seeds[0]:
            #print(model)
            print("# Params:", get_n_params(model))

        # # 对于 IMDB 数据集，使用二元交叉熵损失 BCEWithLogitsLoss。对于其他数据集，使用交叉熵损失 CrossEntropyLoss。
        if args.dataset == 'IMDB':
            loss_fcn = nn.BCEWithLogitsLoss()
        else:
            loss_fcn = nn.CrossEntropyLoss()

        # 分离alpha参数和其他参数
        alpha_params = [p for n, p in model.named_parameters() if n == "alpha"]
        other_params = [p for n, p in model.named_parameters() if n != "alpha"]

        optimizer_w = torch.optim.Adam(other_params, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(alpha_params, lr=args.alr, weight_decay=args.weight_decay)

        train_times = []


        # 开始训练循环，迭代 epochs 次。tqdm 用于显示进度条。
        pbar = tqdm(range(epochs), desc="Training Epochs")
        for epoch in pbar:
            # 手动调用垃圾回收器，确保没有未清理的内存。然后同步 CUDA 设备，记录当前时间。
            gc.collect()
            if not args.cpu: torch.cuda.synchronize()

            eps = 1 - epoch / (epochs - 1)

            # === 每隔N个epoch做一次因果筛选 ===
            if epoch % 10 == 0 and epoch > 0:
                causal_selected_indices = select_causal_metapaths(
                    model, labels, loss_fcn, eval_loader, device,
                    train_val_point, val_test_point,
                    eps, args.ns
                )

                # 若有筛选结果，则采样时只用causal_selected_indices
                if causal_selected_indices is not None:
                    epoch_sampled = causal_selected_indices
                    meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats) if i in epoch_sampled]
                    label_meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats, model.num_paths) if i in epoch_sampled]
                    epoch_feats = {k: v for k, v in feats.items() if k in meta_path_sampled or (model.residual and k == model.tgt_key)}
                    epoch_label_feats = {k: v for k, v in label_feats.items() if k in label_meta_path_sampled}
                    print(f"[Causal Selection] Epoch {epoch}, selected metapath indices: {meta_path_sampled}")
            else:
                epoch_sampled = model.epoch_sample(eps)
                meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats) if i in epoch_sampled]
                label_meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats,model.num_paths) if i in epoch_sampled]
                epoch_feats = {k:v for k,v in feats.items() if k in meta_path_sampled or (model.residual and k==model.tgt_key)}
                epoch_label_feats = {k:v for k,v in label_feats.items() if k in label_meta_path_sampled}



            start = time.time()
            model.set_tau(args.tau_max - (args.tau_max - args.tau_min) * epoch / (epochs - 1))

            # 调用 train 函数进行一次训练，返回损失和准确率。
            train_loss,train_acc,train_micro_f1,train_macro_f1,val_loss,val_acc,val_micro_f1,val_macro_f1 = train_search(model, epoch_feats, epoch_label_feats, labels_cuda,
                                                            loss_fcn,
                                                            optimizer_w, optimizer_a, train_loader, val_loader,
                                                            epoch_sampled, meta_path_sampled, label_meta_path_sampled,
                                                            score, scalar=scalar)
            # # 打印alpha的梯度
            # if model.alpha.grad is not None:
            #     print(f"alpha grad: {model.alpha.grad}")
            # else:
            #     print("alpha grad: None")
            if not args.cpu: torch.cuda.synchronize()
            end = time.time()
            if not args.cpu: torch.cuda.empty_cache()
            train_times.append(end-start)

            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_acc': f'{train_acc:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_acc:.4f}'
            })

            if epoch % 10 == 0:
                alpha_values = model.alpha.detach().cpu().numpy()
                print(f"Epoch {epoch}: alpha values for all meta-paths:")
                for idx, alpha in enumerate(alpha_values):
                    meta_path = model.all_meta_path[idx]  # 获取元路径的具体名称
                    print(f"  Meta-path {idx}: {meta_path}, alpha = {alpha:.4f}")

                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Train Micro-F1: {train_micro_f1 * 100:.4f}, Train Macro-F1: {train_macro_f1 * 100:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                    f"Val Micro-F1: {val_micro_f1 * 100:.4f}, Val Macro-F1: {val_macro_f1 * 100:.4f}"
                )

        ##################direct evaluation##############
        # === 写入最优元路径到arch.py ===
        # 1. 重新用select_causal_metapaths选出最终最优元路径索引
        eps = 0  # 选最优的alpha
        best_indices = select_causal_metapaths(
            model, labels, loss_fcn, eval_loader, device,
            train_val_point, val_test_point,
            eps, args.ns
        )
        # 2. 区分普通元路径和标签元路径
        all_meta_path = list(model.all_meta_path)
        num_feats = model.num_feats
        best_meta = [all_meta_path[i] for i in best_indices if i < num_feats]
        best_label = [all_meta_path[i] for i in best_indices if i >= num_feats]

        print('average train times', sum(train_times) / len(train_times))

        # 3. 写入arch.py
        write_file('arch.py', args.dataset.upper(), best_meta, best_label)

    return best_meta, best_label




def run_study(args):
    results = []

    g, adjs, init_labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full = data_prepare(args)

    # 这句代码的整体作用是将邻接矩阵中每条边的权重重新设置为该边起始节点度的倒数。这样做的目的是进行归一化处理，使得每个节点连接的边权重之和为1，通常用于消息传递中的归一化步骤。
    adjs = adjs_process(adjs)

    # 对节点的索引重新赋值（由于可能去掉了一些没有标签的节点，所以需要对节点的索引进行重新赋值）
    labels, init2sort, sort2init, train_node_nums, total_num_nodes, train_val_point, val_test_point = index_prepare(train_nid,val_nid,test_nid,args.dataset,dl,init_labels)

    # 根据数据集的不同，设置目标节点类型tgt_type、所有节点类型node_types和额外元路径extra_metapath。
    tgt_type, extra_metapath = get_target_info(args.dataset, args.num_hops)

    # 打印当前特征传播的跳数args.num_hops。
    print(f'Current num hops = {args.num_hops}')

    # 平均聚合，进行k跳特征传播
    feats, data_size = neighbour_aggregation(args, g, extra_metapath, tgt_type, init2sort)

    # 设置检查点文件夹
    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)

    # 如果 args.amp 为真，则初始化 GradScaler，用于混合精度训练。
    if args.amp:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    # 对于非 IMDB 数据集，将标签转换为 long 类型，并移动到选定设备。
    # 对于 IMDB 数据集，将标签转换为 float 类型，并移动到选定设备。
    if args.dataset != 'IMDB':
        labels_cuda = labels.long().to(args.device)
    else:
        labels = labels.float()
        labels_cuda = labels.to(args.device)

    # 目标节点数量
    num_target_nodes = dl.nodes['count'][0]

    # 开始训练
    for seed in args.seeds:
        set_random_seed(seed)
        epochs = args.num_epochs

        stopper = EarlyStopping(checkpt_folder,patience=args.patience)

        # 标签特征传播
        label_feats = label_feat_propagate(args, num_target_nodes,tgt_type,num_classes,init_labels,train_nid,val_nid,test_nid,init2sort,adjs)

        # 创建训练数据加载器 train_loader，随机打乱训练节点索引，并设定 batch_size 和 drop_last 参数。
        train_loader = torch.utils.data.DataLoader(
            torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)

        # 评估数据加载器 eval_loader 和多余数据加载器 full_loader。
        eval_loader, full_loader = add_eval_loader(args.batch_size, total_num_nodes, num_target_nodes, feats, label_feats)

        # 释放显存和手动触发垃圾回收。
        torch.cuda.empty_cache()
        gc.collect()

        # todo 将path改成feats.keys()
        model = CMS(args.embed_size, args.hidden_size, num_classes, feats.keys(), label_feats.keys(), tgt_type,
            args.dropout, args.input_drop, args.att_drop, args.label_drop,
                      args.env_type, args.env_layer_number, args.tau, args.K,
            args.n_layers_2,  args.residual, bns=args.bns, data_size=data_size, path=feats.keys(),
            label_path=label_feats.keys(), eps=args.eps, device=args.device)

        model = model.to(args.device)

        # 打印模型参数总数（在 args.seed 为初始值时）。
        if seed == args.seeds[0]:
            #print(model)
            print("# Params:", get_n_params(model))

        # # 对于 IMDB 数据集，使用二元交叉熵损失 BCEWithLogitsLoss。对于其他数据集，使用交叉熵损失 CrossEntropyLoss。
        if args.dataset == 'IMDB':
            loss_fcn = nn.BCEWithLogitsLoss()
        else:
            loss_fcn = nn.CrossEntropyLoss()
        # 创建 Adam 优化器，使用模型的参数，并设置学习率和权重衰减。
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

        train_times = []


        # 开始训练循环，迭代 epochs 次。tqdm 用于显示进度条。
        pbar = tqdm(range(epochs), desc="Training Epochs")
        for epoch in pbar:
            # 手动调用垃圾回收器，确保没有未清理的内存。然后同步 CUDA 设备，记录当前时间。
            gc.collect()
            if not args.cpu: torch.cuda.synchronize()
            start = time.time()

            # 调用 train 函数进行一次训练，返回损失和准确率。
            loss, accuracy, micro_f1, macro_f1 = train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, score, args.lamda, scalar=scalar)

            # 再次同步 CUDA 设备，记录训练结束的时间。
            if not args.cpu: torch.cuda.synchronize()
            end = time.time()
            train_time = end - start
            # 将本轮训练所需的时间添加到 train_times 列表中。
            train_times.append(train_time)

            log = "Epoch {}, training Time(s): {:.4f}, estimated train loss {:.4f}, accuracy {:.4f}, Micro f1 {:.4f}, Macro f1 {:.4f}\n".format(epoch, train_time, loss, accuracy, micro_f1 * 100, macro_f1 * 100)
            # 更新tqdm进度条的后缀信息，包含训练集和验证集的所有指标
            pbar.set_postfix({
                'train_loss': f'{loss:.4f}',
                'train_acc': f'{accuracy:.4f}',
                'train_micro_f1': f'{micro_f1:.4f}',
                'train_macro_f1': f'{macro_f1:.4f}'
            })

            # 仅在检测到显存紧张后清空
            if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.8:
                torch.cuda.empty_cache()

            # 在验证集和测试集上进行评估
            # todo 待优化，这里只影响性能，可以改成每轮只在验证集上进行评估，每当出现一个更好的验证集的时候，再在测试集上进行评估
            val_loss, test_loss, train_accuracy, train_micro_f1, train_macro_f1, val_accuracy, val_micro_f1, val_macro_f1, test_accuracy, test_micro_f1, test_macro_f1, raw_preds = evaluate(
                model,args.dataset,args.device,eval_loader,loss_fcn,labels,total_num_nodes,train_val_point,val_test_point)


            # log += f'evaluation Time: {end-start}, Train loss: {loss_train}, Val loss: {loss_val}, Test loss: {loss_test}\n'
            log += f'Val loss: {val_loss}, Test loss: {test_loss}\n'
            log += 'Train: (micro-f1: {:.4f}, macro-f1: {:.4f}), Val: (micro-f1: {:.4f}, macro-f1: {:.4f}), Test acc: (micro-f1: {:.4f}, macro-f1: {:.4f}) ({})\n'.format(
                train_micro_f1 * 100, train_macro_f1 * 100, val_micro_f1 * 100, val_macro_f1 * 100, test_micro_f1 * 100,
                test_macro_f1 * 100, total_num_nodes - val_test_point)

            early_stop = stopper.step(val_loss, val_accuracy, val_micro_f1, val_macro_f1, epoch, raw_preds, model)

            # 每隔 10 轮输出当前最佳验证损失
            if epoch > 0 and epoch % 10 == 0:
                log += f'\tCurrent best at epoch {stopper.best_epoch} with Val loss:{stopper.best_loss:.4f}, acc:{stopper.best_acc:.4f}, micro-f1:{stopper.best_micro_f1 * 100:.4f}, macro-f1:{stopper.best_macro_f1 * 100:.4f}'
            # 控制日志输出
            if epoch % 5 == 0:
                print(log)

            if early_stop:
                break

        best_pred = stopper.best_pred
        # 如果有 full_loader，则加载保存的最佳模型，进行评估并生成最终预测。
        best_pred = full_evaluate(model, stopper, best_pred, args.device, full_loader)

        # 根据数据集类型生成最终预测概率并生成输出文件，保存预测结果以供后续评估。
        if args.dataset != 'IMDB':
            predict_prob = best_pred.softmax(dim=1)
        else:
            predict_prob = torch.sigmoid(best_pred)

        test_logits = predict_prob[sort2init][test_nid_full]
        if args.dataset != 'IMDB':
            pred = test_logits.cpu().numpy().argmax(axis=1)
            dl.gen_file_for_evaluate(test_idx=test_nid_full, label=pred,
                                     file_path=f"./output/{args.dataset}_{seed}_{os.path.splitext(stopper.filename.split('/')[-1])[0]}.txt")
        else:
            pred = (test_logits.cpu().numpy() > 0.5).astype(int)
            dl.gen_file_for_evaluate(test_idx=test_nid_full, label=pred,
                                     file_path=f"./output/{args.dataset}_{seed}_{os.path.splitext(stopper.filename.split('/')[-1])[0]}.txt",
                                     mode='multi')

        # 计算并打印训练、验证和测试集的准确率，以便评估模型在各个数据集上的表现。
        if args.dataset != 'IMDB':
            preds = predict_prob.argmax(dim=1, keepdim=True)
        else:
            preds = (predict_prob > 0.5).int()
        train_accuracy, train_micro_f1, train_macro_f1 = score(labels[:train_val_point], preds[:train_val_point])
        val_accuracy, val_micro_f1, val_macro_f1 = score(labels[train_val_point:val_test_point], preds[train_val_point:val_test_point])
        test_accuracy, test_micro_f1, test_macro_f1 = score(labels[val_test_point:total_num_nodes], preds[val_test_point:total_num_nodes])

        print(f'train acc: {train_accuracy:.2f}, f1: ({train_micro_f1*100:.2f}, {train_macro_f1*100:.2f}) ' \
            + f'val acc: {val_accuracy:.2f}, f1: ({val_micro_f1*100:.2f}, {val_macro_f1*100:.2f}) ' \
            + f'test acc: {test_accuracy:.2f}, f1: ({test_micro_f1*100:.2f}, {test_macro_f1*100:.2f})')

        # 删除保存的最优的模型
        os.remove(stopper.filename)

        results.append([test_micro_f1 * 100, test_macro_f1 * 100])

    if args.dataset == 'IMDB':
        results.sort(key=lambda x: x[0], reverse=True)
        print('Experimental Results:')
        print(results)
        results = results[:5]
        mima = list(map(list, zip(*results)))
        #print(f'micro: {mima[0]}', f'macro: {mima[1]}')
        print(f'micro_mean: {np.mean(mima[0]):.2f}', f'micro_std: {np.std(mima[0]):.2f}')
        print(f'macro_mean: {np.mean(mima[1]):.2f}', f'macro_std: {np.std(mima[1]):.2f}')
    else:
        aver = list(map(list, zip(*results)))
        print(f'micro_aver: {np.mean(aver[0]):.2f}', f'micro_std: {np.std(aver[0]):.2f}')
        print(f'macro_aver: {np.mean(aver[1]):.2f}', f'macro_std: {np.std(aver[1]):.2f}')

    if args.dataset == 'IMDB':
        results.sort(key=lambda x: x[0], reverse=True)
        results = results[:5]
        mima = list(map(list, zip(*results)))
        return {
            "micro_mean": round(np.mean(mima[0]), 2),
            "micro_std": round(np.std(mima[0]), 2),
            "macro_mean": round(np.mean(mima[1]), 2),
            "macro_std": round(np.std(mima[1]), 2),
        }
    else:
        aver = list(map(list, zip(*results)))
        return {
            "micro_aver": round(np.mean(aver[0]), 2),
            "micro_std": round(np.std(aver[0]), 2),
            "macro_aver": round(np.mean(aver[1]), 2),
            "macro_std": round(np.std(aver[1]), 2),
        }
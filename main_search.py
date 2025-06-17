import torch
import os
import gc
import time
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from early_stopping import EarlyStopping
from utils import set_random_seed, get_n_params, select_causal_metapaths, train_search, write_file
from params import parse_args_search
from evaluate_util import score, evaluate, full_evaluate, self_f1_score
from data_prepare import data_prepare, index_prepare, get_target_info, add_eval_loader
from data_process import adjs_process, neighbour_aggregation, label_feat_propagate
from mdoel_search import CMS_Se

def main(args):

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

            # 论文未发表，暂时隐藏



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



if __name__ == "__main__":
    args = parse_args_search()

    device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'
    args.device = device

    main(args)
import datetime
import errno
import os
import random
import torch.nn as nn
import ast
import numpy as np
import torch

from evaluate_util import eval_with_indices


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print("Directory {} already exists.".format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = "{}_{:02d}-{:02d}-{:02d}".format(
        dt.date(), dt.hour, dt.minute, dt.second
    )

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args["log_dir"], "{}_{}".format(args["dataset"], date_postfix)
    )

    if sampling:
        log_dir = log_dir + "_sampling"

    mkdir_p(log_dir)
    return log_dir


def parse_meta_paths(meta_path_str):
    # 每两个字符为一个节点类型
    nodes = list(meta_path_str)
    edge_types = []
    for i in range(len(nodes) - 1):
        edge_types.append(f"{nodes[i]}-{nodes[i+1]}")
    return edge_types

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, lamda, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        ## batch = batch.to(device)
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att, reg = model(batch, batch_feats, batch_labels_feats)
                loss_train = loss_fcn(output_att, batch_y)

                # 在训练模式下，将 reg 加到总损失中
                if model.training:
                    loss_train += reg * lamda

            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att, reg = model(batch, batch_feats, batch_labels_feats)
            L1 = loss_fcn(output_att, batch_y)

            # 在训练模式下，将 reg 加到总损失中
            if model.training:
                L1 += reg * lamda

            loss_train = L1
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    accuracy, micro_f1, macro_f1 = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, accuracy, micro_f1, macro_f1


def select_causal_metapaths(model, labels, loss_fcn, eval_loader, device, train_val_point, val_test_point, eps, num_sampled):
    # 论文未发表，暂时隐藏
    return ""


def train_search(model, feats, label_feats, labels_cuda, loss_fcn, optimizer_w, optimizer_a, train_loader, val_loader, epoch_sampled, meta_path_sampled, label_meta_path_sampled, evaluator, scalar=None):
    model.train()
    device = labels_cuda.device

    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []
    val_total_loss = 0
    val_y_true, val_y_pred = [], []
    ###################  optimize w  ##################
    # 开始训练循环（优化 w）
    for batch in train_loader:
        # batch = batch.to(device)
        # 从验证集中取一个 batch 作为对应监督信号（优化 a）
        # 为每个训练 batch 都取一个验证 batch 来更新结构参数（a）——一种交替优化策略。
        val_batch = next(iter(val_loader))
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
            val_batch_feats = [x[val_batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            # import code
            # code.interact(local=locals())
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            val_batch_feats = {k: x[val_batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        val_batch_labels_feats = {k: x[val_batch].to(device) for k, x in label_feats.items()}

        # 取出当前 batch 的标签
        batch_y = labels_cuda[batch]
        val_batch_y = labels_cuda[val_batch]

        ########################################train
        # 论文未发表，暂时隐藏


        ########################################
        # 记录指标
        y_true.append(batch_y.cpu().to(torch.long))
        val_y_true.append(val_batch_y.cpu().to(torch.long))
        # 收集预测结果：对多分类用 argmax，二分类用 > 0.
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
            val_y_pred.append((val_output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            val_y_pred.append(val_output_att.argmax(dim=-1, keepdim=True).cpu())

        # 计算整体损失与精度
        total_loss += loss_train.item()
        val_total_loss += val_loss_train.item()
        iter_num += 1


    train_loss = total_loss / iter_num
    train_acc,train_micro_f1,train_macro_f1 = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    val_loss = val_total_loss / iter_num
    val_acc,val_micro_f1,val_macro_f1 = evaluator(torch.cat(val_y_true, dim=0), torch.cat(val_y_pred, dim=0))

    return train_loss,train_acc,train_micro_f1,train_macro_f1,val_loss,val_acc,val_micro_f1,val_macro_f1


# 写入arch.py
def write_file(write_file_name, dataset, best_path, best_label_path):
    arch_path = os.path.join(os.path.dirname(__file__), write_file_name)
    with open(arch_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 提取archs字典体
    start = content.find('archs = {')
    end = content.rfind('}')
    archs_body = content[start+len('archs = {'):end]
    # 用ast安全解析
    archs_dict = ast.literal_eval('{' + archs_body + '}')
    # 更新
    archs_dict[dataset] = [best_path, best_label_path]
    # 重新写回
    with open(arch_path, 'w', encoding='utf-8') as f:
        f.write('\narchs = {\n')
        for k, v in archs_dict.items():
            f.write(f'    "{k}": {v},\n')
        f.write('}\n')
    print(f"最优元路径已写入{write_file_name}: {best_path}, 标签元路径: {best_label_path}")
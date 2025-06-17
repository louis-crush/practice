import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def self_f1_score(labels, prediction):
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")
    return micro_f1, macro_f1

def score(labels, prediction):
    labels = labels.cpu().squeeze()
    prediction = prediction.cpu().squeeze()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1, macro_f1 = self_f1_score(labels, prediction)

    return accuracy, micro_f1, macro_f1


def evaluate(model, dataset, device, eval_loader, loss_fcn, labels, total_num_nodes, train_val_point, val_test_point):
    labels = labels.to(device)
    # 在验证集和测试集上评估模型性能。
    # 记录验证损失和测试损失，并基于它们更新最佳模型。
    # 禁用梯度计算以提高评估速度，将模型设置为评估模式，并初始化 raw_preds 列表用于存储预测结果。
    # todo 待修改，在gpu上运行
    with torch.no_grad():
        model.eval()
        raw_preds = []

        # 对于每个评估批次，将数据转移到设备并执行模型的前向传播，生成预测结果并将其添加到 raw_preds 列表。
        for batch, batch_feats, batch_labels_feats in eval_loader:
            batch = batch.to(device)
            batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
            batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
            raw_preds.append(model(batch, batch_feats, batch_labels_feats, False))

        # 将所有批次的预测结果合并为一个张量。
        raw_preds = torch.cat(raw_preds, dim=0)
        # 计算训练、验证和测试损失。
        # train_loss = loss_fcn(raw_preds[:train_val_point], labels[:train_val_point]).item()
        val_loss = loss_fcn(raw_preds[train_val_point:val_test_point],
                            labels[train_val_point:val_test_point]).item()
        test_loss = loss_fcn(raw_preds[val_test_point:total_num_nodes],
                             labels[val_test_point:total_num_nodes]).item()

    # 根据数据集类型，生成最终预测结果。
    if dataset != 'IMDB':
        preds = raw_preds.argmax(dim=-1)
    else:
        preds = (raw_preds > 0.).int()

    # 使用 evaluator 计算训练、验证和测试集的准确率。
    train_accuracy, train_micro_f1, train_macro_f1 = score(preds[:train_val_point], labels[:train_val_point])
    val_accuracy, val_micro_f1, val_macro_f1 = score(preds[train_val_point:val_test_point], labels[train_val_point:val_test_point])
    test_accuracy, test_micro_f1, test_macro_f1 = score(preds[val_test_point:total_num_nodes], labels[val_test_point:total_num_nodes])

    return val_loss, test_loss, train_accuracy, train_micro_f1, train_macro_f1, val_accuracy, val_micro_f1, val_macro_f1, test_accuracy, test_micro_f1, test_macro_f1, raw_preds


# 这个方法用于计算正则化损失
# 正则化：通过计算 Gumbel-Softmax 或 Softmax 编码 z 与 log-softmax log_pi 的加权对数概率，可以对模型的环境编码进行正则化，防止过拟合。
# 信息熵：这种计算方式类似于信息熵，可以衡量环境编码的分布情况，鼓励模型在不同环境之间进行适当的权衡和选择。
# 数值稳定性：使用 log-sum-exp 技术计算 log-softmax，可以提高数值稳定性，防止计算过程中的溢出或下溢。
def reg_loss(z, logit, logit_0=None):
    log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, 1, logit.size(2))
    return torch.mean(torch.sum(torch.mul(z, log_pi)))


def full_evaluate(model, stopper, best_pred, device, full_loader):
    # 论文未发表，暂时隐藏
    return ""


def eval_with_indices():
    # 论文未发表，暂时隐藏
    return ""
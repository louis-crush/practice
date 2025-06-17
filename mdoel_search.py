import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

import random


class CMS_Se(nn.Module):

    # hidden: 隐藏层的维度。
    # nclass: 分类的类别数。
    # feat_keys: 特征键的列表。
    # label_feat_keys: 标签特征键的列表。
    # tgt_key: 目标键，通常是需要进行预测的特征。
    # dropout, input_drop: 丢弃率，用于防止过拟合。
    # device: 设备信息（如 GPU）。
    # num_final: 最终保留的路径数量。
    # residual: 是否使用残差连接。
    # bns: 是否使用批归一化。
    # data_size: 数据大小，用于初始化嵌入。
    # num_sampled: 每次采样的数量。
    def __init__(self, hidden, nclass, feat_keys, label_feat_keys, tgt_key, dropout,
                 input_drop, device, num_final, residual=False, bns=False, data_size=None, num_sampled=1):

        super(CMS_Se, self).__init__()
        # 初始化特征和标签特征的键，以及路径的数量和样本数量等。
        self.feat_keys = feat_keys
        self.label_feat_keys = label_feat_keys
        self.num_feats = len(feat_keys)
        self.all_meta_path = list(self.feat_keys) + list(self.label_feat_keys)
        self.num_sampled = num_sampled
        self.num_channels = self.num_sampled
        self.num_paths = len(self.all_meta_path)
        self.num_final = num_final
        self.num_res = self.num_paths - self.num_final
        self.tgt_key = tgt_key
        self.residual = residual


        self.init_params()

        # 论文未发表，暂时隐藏



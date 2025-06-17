import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch.nn.parameter import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

from evaluate_util import reg_loss





class CMS(nn.Module):
    # 主要参数包括特征维度 nfeat、隐藏层维度 hidden、类别数 nclass 等，还有多个丢弃率和标签处理参数。
    def __init__(self, nfeat, hidden, nclass, feat_keys, label_feat_keys, tgt_type,
                 dropout, input_drop, att_dropout, label_drop,
                 env_type, env_layer_number, tau, K,
                 n_layers_2, residual=False, bns=False, data_size=None, path=[],
                 label_path=[], eps=0, device=None):
        super(CMS, self).__init__()

        # feat_keys 和 label_feat_keys 分别表示普通特征和标签特征的键列表。
        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        # num_channels 是网络的通道数，为 path 和 label_path 的长度和。
        self.num_channels = len(path) + len(label_path)
        # tgt_type 表示目标类型，residual 表示是否使用残差连接。
        self.tgt_type = tgt_type
        self.residual = residual

        # 存储 data_size 数据集大小和 path、label_path 信息。
        self.data_size = data_size
        self.path = path
        self.label_path = label_path
        # self.embeding 是一个字典，用于存储不同特征的嵌入。
        self.embeding = nn.ParameterDict({})

        # 论文未发表，暂时隐藏


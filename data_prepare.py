import torch
import numpy as np
import dgl
import datetime

from torch_sparse import SparseTensor

from data_loader import data_loader

def data_prepare(args):
    dl = data_loader(f'{args.root}/{args.dataset}')

    # === feats ===
    # 每种节点类型的特征
    features_list = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features_list.append(torch.eye(dl.nodes['count'][i]))
        else:
            features_list.append(torch.FloatTensor(th))

    # === labels ===
    num_classes = dl.labels_train['num_classes']
    # 为什么节点数是dl.nodes['count'][0]
    # 因为0类型才是目标节点
    init_labels = np.zeros((dl.nodes['count'][0], num_classes), dtype=int)

    # 获取训练接和验证集的节点索引
    # 验证集的节点占所有节点的比例
    val_ratio = 0.2
    # 获取所有有标签的节点索引
    # np.nonzero(...)：返回所有 True 值的索引。
    # 形式是一个元组：(array([i1, i2, i3, ...]),)
    # [0]：从元组中取出第一个维度的索引数组。
    train_nid = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_nid)
    split = int(train_nid.shape[0] * val_ratio)
    val_nid = train_nid[:split]
    train_nid = train_nid[split:]
    train_nid = np.sort(train_nid)
    val_nid = np.sort(val_nid)

    # 获取测试集的节点索引
    test_nid = np.nonzero(dl.labels_test['mask'])[0]
    test_nid_full = np.nonzero(dl.labels_test_full['mask'])[0]

    init_labels[train_nid] = dl.labels_train['data'][train_nid]
    init_labels[val_nid] = dl.labels_train['data'][val_nid]
    init_labels[test_nid] = dl.labels_test['data'][test_nid]
    # IMDB是多标签分类，其他都是单标签
    if args.dataset != 'IMDB':
        init_labels = init_labels.argmax(axis=1)

    print(len(train_nid), len(val_nid), len(test_nid), len(test_nid_full))
    init_labels = torch.LongTensor(init_labels)

    # 每种节点类型在异构图中节点索引的起始位置
    idx_shift = np.zeros(len(dl.nodes['count']) + 1, dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        idx_shift[i + 1] = idx_shift[i] + dl.nodes['count'][i]

    # === adjs ===
    # todo 暂不考虑Freebase
    adjs = []
    for i, (k, v) in enumerate(dl.links['data'].items()):
        v = v.tocoo()
        # 根据 v.col[0] 和 v.row[0] 的节点索引，推断它们各自属于哪个节点类型
        src_type_idx = np.where(idx_shift > v.col[0])[0][0] - 1
        dst_type_idx = np.where(idx_shift > v.row[0])[0][0] - 1
        # 将全局节点编号转换为局部编号
        row = v.row - idx_shift[dst_type_idx]
        col = v.col - idx_shift[src_type_idx]
        # 通过使用这些索引从 dl.nodes['count'] 中提取相应的节点数量，sparse_sizes 变成一个元组 (源节点类型数量, 目标节点类型数量)。
        sparse_sizes = (dl.nodes['count'][dst_type_idx], dl.nodes['count'][src_type_idx])
        # 创建邻接矩阵
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)
        adjs.append(adj)

    if args.dataset == 'DBLP':
        # A* --- P --- T
        #        |
        #        V
        # author: [4057, 334]
        # paper : [14328, 4231]
        # term  : [7723, 50]
        # venue(conference) : None
        A, P, T, V = features_list
        AP, PA, PT, PV, TP, VP = adjs

        new_edges = {}
        etypes = [  # src->tgt
            ('P', 'P-A', 'A'),
            ('A', 'A-P', 'P'),
            ('T', 'T-P', 'P'),
            ('V', 'V-P', 'P'),
            ('P', 'P-T', 'T'),
            ('P', 'P-V', 'V'),
        ]

        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
        g = dgl.heterograph(new_edges)

        # for i, etype in enumerate(g.etypes):
        #     src, dst, eid = g._graph.edges(i)
        #     adj = SparseTensor(row=dst.long(), col=src.long())
        #     print(etype, adj)

        # g.ndata['feat']['A'] = A # not work
        g.nodes['A'].data['A'] = A
        g.nodes['P'].data['P'] = P
        g.nodes['T'].data['T'] = T
        g.nodes['V'].data['V'] = V
    elif args.dataset == 'IMDB':
        # A --- M* --- D
        #       |
        #       K
        # movie    : [4932, 3489]
        # director : [2393, 3341]
        # actor    : [6124, 3341]
        # keywords : None
        M, D, A, K = features_list
        MD, DM, MA, AM, MK, KM = adjs
        assert torch.all(DM.storage.col() == MD.t().storage.col())
        assert torch.all(AM.storage.col() == MA.t().storage.col())
        assert torch.all(KM.storage.col() == MK.t().storage.col())

        assert torch.all(MD.storage.rowcount() == 1)  # each movie has single director

        new_edges = {}
        etypes = [  # src->tgt
            ('D', 'D-M', 'M'),
            ('M', 'M-D', 'D'),
            ('A', 'A-M', 'M'),
            ('M', 'M-A', 'A'),
            ('K', 'K-M', 'M'),
            ('M', 'M-K', 'K'),
        ]

        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)

        g = dgl.heterograph(new_edges)

        g.nodes['M'].data['M'] = M
        g.nodes['D'].data['D'] = D
        g.nodes['A'].data['A'] = A
        if args.num_hops > 2:  # or args.two_layer:
            g.nodes['K'].data['K'] = K
    elif args.dataset == 'ACM':
        # A --- P* --- C
        #       |
        #       K
        # paper     : [3025, 1902]
        # author    : [5959, 1902]
        # conference: [56, 1902]
        # field     : None
        P, A, C, K = features_list
        PP, PP_r, PA, AP, PC, CP, PK, KP = adjs
        row, col = torch.where(P)
        assert torch.all(row == PK.storage.row()) and torch.all(col == PK.storage.col())
        assert torch.all(AP.matmul(PK).to_dense() == A)
        assert torch.all(CP.matmul(PK).to_dense() == C)

        assert torch.all(PA.storage.col() == AP.t().storage.col())
        assert torch.all(PC.storage.col() == CP.t().storage.col())
        assert torch.all(PK.storage.col() == KP.t().storage.col())

        row0, col0, _ = PP.coo()
        row1, col1, _ = PP_r.coo()
        PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=PP.sparse_sizes())
        PP = PP.coalesce()
        PP = PP.set_diag()
        adjs = [PP] + adjs[2:]

        new_edges = {}
        etypes = [  # src->tgt
            ('P', 'P-P', 'P'),
            ('A', 'A-P', 'P'),
            ('P', 'P-A', 'A'),
            ('C', 'C-P', 'P'),
            ('P', 'P-C', 'C'),
        ]

        if args.ACM_keep_F:
            etypes += [
                ('K', 'K-P', 'P'),
                ('P', 'P-K', 'K'),
            ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)

        g = dgl.heterograph(new_edges)

        g.nodes['P'].data['P'] = P  # [3025, 1902]
        g.nodes['A'].data['A'] = A  # [5959, 1902]
        g.nodes['C'].data['C'] = C  # [56, 1902]
        if args.ACM_keep_F:
            g.nodes['K'].data['K'] = K  # [1902, 1902]
    else:
        assert 0

    if args.dataset == 'DBLP':
        adjs = {'AP': AP, 'PA': PA, 'PT': PT, 'PV': PV, 'TP': TP, 'VP': VP}
    elif args.dataset == 'ACM':
        adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP}
    elif args.dataset == 'IMDB':
        adjs = {'MD': MD, 'DM': DM, 'MA': MA, 'AM': AM, 'MK': MK, 'KM': KM}
    else:
        assert 0

    return g, adjs, init_labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full


# 重新分配节点索引
def index_prepare(train_nid, val_nid, test_nid, dataset, dl, init_labels):
    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)

    # 设置训练和验证分割点
    train_val_point = train_node_nums
    # 设置验证和测试分割点
    val_test_point = train_val_point + valid_node_nums
    # 计算训练、验证和测试节点的总数
    total_num_nodes = train_node_nums + valid_node_nums + test_node_nums

    #目标节点数量
    num_target_nodes = dl.nodes['count'][0]

    # todo 不确定哪个数据集会有用
    # 检查 total_num_nodes 是否小于 num_target_nodes，如果用于学习的总节点数量小于目标节点的数量，则表示有多余节点。
    # # 检查是否有额外的未标注节点，如果存在，则记录这些节点。
    if total_num_nodes < num_target_nodes:
        # 创建一个布尔数组 flag，初始化为全 True，表示假设所有节点都是多余的。
        flag = np.ones(num_target_nodes, dtype=bool)
        # 将训练节点的索引对应位置设为 False，表示这些节点不是多余的。
        flag[train_nid] = 0
        # 将验证节点索引对应位置设为 False。
        flag[val_nid] = 0
        # 将测试节点索引对应位置设为 False。
        flag[test_nid] = 0
        # 获取 flag 数组中值为 True 的位置索引，这些即为多余节点的索引。
        extra_nid = np.where(flag)[0]
        # 打印出找到的多余节点数量 len(extra_nid)。
        print(f'Find {len(extra_nid)} extra nid for dataset {dataset}')
    else:
        # 如果 total_num_nodes 大于等于 num_target_nodes，则没有多余节点，将 extra_nid 设置为空数组
        extra_nid = np.array([])

    # init2sort 是一个 torch.LongTensor 类型的张量，将 train_nid、val_nid、test_nid 和 extra_nid 连接在一起，生成了包含所有节点的一个序列，顺序是训练、验证、测试和多余节点。
    init2sort = torch.LongTensor(np.concatenate([train_nid, val_nid, test_nid, extra_nid]))
    # sort2init 表示原始索引到排序索引的映射关系。
    # torch.argsort(init2sort) 返回 init2sort 排序后对应的原始索引位置
    sort2init = torch.argsort(init2sort)
    # 断言语句用于验证重新排列后的标签与初始标签 init_labels 相同。
    # init_labels[init2sort][sort2init] 先按 init2sort 的顺序提取标签，然后按 sort2init 的顺序重新排列，还原成 init_labels，以确保顺序恢复后数据没有发生改变。
    assert torch.all(init_labels[init2sort][sort2init] == init_labels)
    # labels 是按 init2sort 排序后的标签，用于模型训练中节点标签的有序输入。
    labels = init_labels[init2sort]

    return labels, init2sort, sort2init, train_node_nums, total_num_nodes, train_val_point, val_test_point

def get_target_info(dataset, num_hops):
    # 根据数据集的不同，设置目标节点类型tgt_type、所有节点类型node_types和额外元路径extra_metapath。
    if dataset == 'DBLP':
        tgt_type = 'A'
        node_types = ['A', 'P', 'T', 'V']
        extra_metapath = []
    elif dataset == 'ACM':
        tgt_type = 'P'
        node_types = ['P', 'A', 'C']
        extra_metapath = []
    elif dataset == 'IMDB':
        tgt_type = 'M'
        node_types = ['M', 'A', 'D', 'K']
        extra_metapath = []
    elif dataset == 'Freebase':
        tgt_type = '0'
        node_types = [str(i) for i in range(8)]
        extra_metapath = []
    else:
        assert 0
    # 过滤extra_metapath，只保留长度大于args.num_hops + 1的路径。
    extra_metapath = [ele for ele in extra_metapath if len(ele) > num_hops + 1]

    return tgt_type, extra_metapath


def add_eval_loader(batch_size, total_num_nodes, num_target_nodes, feats, label_feats):
    eval_loader, full_loader = [], []
    # 设置 batchsize 为 args.batch_size 的两倍。
    # batchsize只在评估时使用，评估时通常希望尽量减少批次数，以加快评估速度（因为不需要反向传播，也不需要太多随机性）。由于评估时不涉及梯度更新，可以适当加大 batch size，以充分利用内存和加速推理。
    # 这里直接用 2 * args.batch_size，是一个经验性做法，既能加快评估，又不会太容易 OOM（超出内存）。
    batchsize = 2 * batch_size

    # 将训练、验证和测试节点分批加载到 eval_loader。
    # 对每个批次索引 batch_idx，计算起止点，将相应特征和标签特征提取并添加到 eval_loader。
    for batch_idx in range((total_num_nodes - 1) // batchsize + 1):
        batch_start = batch_idx * batchsize
        batch_end = min(total_num_nodes, (batch_idx + 1) * batchsize)
        batch = torch.arange(batch_start, batch_end)

        batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
        batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}

        eval_loader.append((batch, batch_feats, batch_labels_feats))

    # 多余数据（total_num_nodes是所有有标签的目标节点，额外节点，就是目标节点中存在没有标签的目标节点）
    for batch_idx in range((num_target_nodes - total_num_nodes - 1) // batchsize + 1):
        batch_start = batch_idx * batchsize + total_num_nodes
        batch_end = min(num_target_nodes, (batch_idx + 1) * batchsize + total_num_nodes)
        batch = torch.arange(batch_start, batch_end)

        batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
        batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}

        full_loader.append((batch, batch_feats, batch_labels_feats))

    return eval_loader,full_loader

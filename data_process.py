import torch
import datetime
import gc
import dgl.function as fn
import torch.nn.functional as F

from torch_sparse import remove_diag
from tqdm import tqdm

from arch import archs

def adjs_process(adjs):
    # 遍历邻接矩阵adjs的每个键值对k
    # 这句代码的整体作用是将邻接矩阵中每条边的权重重新设置为该边起始节点度的倒数。这样做的目的是进行归一化处理，使得每个节点连接的边权重之和为1，通常用于消息传递中的归一化步骤。
    # todo 这里不知道对于非 Freebase 数据集是否有作用，待检查（对于非 Freebase 数据集，使用列表存储；对于 Freebase 数据集，使用字典存储。）
    for k in adjs.keys():
        # 清除当前邻接矩阵adjs[k]的权重值。
        adjs[k].storage._value = None
        # 设置新的权重值，所有边的权重设置为平均值。
        # adjs[k] 是一个稀疏矩阵（通常是 DGLGraph 的邻接矩阵）。
        # nnz() 函数返回稀疏矩阵中非零元素的数量（number of non-zero elements）。
        # torch.ones 函数创建一个包含全为1的张量，其长度为 adjs[k].nnz()，即包含和 adjs[k] 中非零元素数量相同的1。
        # 计算邻接矩阵 adjs[k] 每行的和，结果是一个一维张量，其中每个元素代表邻接矩阵对应行的和。
        # dim=-1 表示在最后一个维度上进行求和，这里指的是行求和。
        # adjs[k].storage 是稀疏矩阵的底层存储结构，row() 返回一个张量，表示稀疏矩阵中每个非零元素的行索引。
        # 通过使用行索引张量 adjs[k].storage.row() 访问按行求和的结果，这会为每个非零元素选择其对应的行和。
        # 结果是一个张量，其长度等于 adjs[k] 中非零元素的数量，每个元素是相应非零元素所在行的和。
        # 对于每个非零元素，计算1除以该元素所在行的和。
        # 结果是一个张量，其长度为 adjs[k].nnz()，每个元素代表邻接矩阵中每条边的权重。
        # 最终，将计算得到的权重张量分配给 adjs[k].storage._value，这将更新邻接矩阵中每条边的权重。
        adjs[k].storage._value = torch.ones(adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]

    return adjs


def neighbour_aggregation(args, g, extra_metapath, tgt_type, init2sort):
    # 记录当前时间，用于计算特征传播所花费的时间。
    prop_tic = datetime.datetime.now()

    # 计算最大特征传播路径长度 max_length。
    # 如果 extra_metapath（额外的元路径）非空，则取 args.num_hops + 1 和 extra_metapath 中最长元路径长度的最大值。
    # 否则，max_length 仅设为 args.num_hops + 1。
    if len(extra_metapath):
        max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
    else:
        max_length = args.num_hops + 1

    # 调用 hg_propagate_feat_dgl_path 函数，对图 g 的目标节点类型 tgt_type 进行特征传播。
    # 传播范围是 args.num_hops。
    # max_length 决定了传播路径的最大长度。
    # archs[args.dataset][0] 表示目标数据集的第一条元路径
    # echo=False 表示在传播过程中不打印中间输出。
    g = hg_propagate_feat_dgl_path(g, tgt_type, args.num_hops, max_length, archs[args.dataset][0], echo=False)

    # 初始化一个空字典 feats，用于存储目标节点类型 tgt_type 的特征。
    feats = {}
    # keys 是 g 中目标节点类型 tgt_type 上的特征键列表。
    keys = list(g.nodes[tgt_type].data.keys())
    # 打印目标节点类型和其特征键，用于调试或检查特征名称。
    print(f'For tgt {tgt_type}, feature keys {keys}')
    # 遍历目标节点的特征键 keys，将每个特征 k 从 g 的数据中移除并存储到 feats 字典中。
    # pop(k) 用于移除特征，避免将不需要的特征留在 g 中，以节省内存。
    for k in keys:
        feats[k] = g.nodes[tgt_type].data.pop(k)

    # 判断数据集是否为 DBLP、ACM 或 IMDB。
    # 如果是，data_size 记录每个特征 v 的维度大小。
    # 然后，feats 中每个特征根据 init2sort 进行重新排序。
    # 如果数据集不是这三个，assert 0 会触发异常并终止程序。
    if args.dataset in ['DBLP', 'ACM', 'IMDB']:
        data_size = {k: v.size(-1) for k, v in feats.items()}
        feats = {k: v[init2sort] for k, v in feats.items()}
    else:
        assert 0

    # 记录当前时间 prop_toc 并计算特征传播花费的时间。
    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    # 调用 gc.collect() 进行垃圾回收，释放内存。这在大规模图数据处理中尤其重要，以防内存泄漏或冗余占用。
    gc.collect()

    return feats, data_size

def hg_propagate_feat_dgl_path(g, tgt_type, num_hops, max_length, meta_path, echo=False):

    # todo 这个可以优化，只传递在meta_path的元路径

    for hop in range(1, max_length):
        #reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]

        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)

            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    # if hop == max_length - 1:
                    #     import code
                    #     code.interact(local=locals())
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type ) \
                      or (hop > num_hops):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

        # remove no-use items
        for ntype in g.ntypes:

            if ntype == tgt_type: continue
            removes = []
            for k in g.nodes[ntype].data.keys():

                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo:
            print(f'-- hop={hop} ---')
            for ntype in g.ntypes:
                for k, v in g.nodes[ntype].data.items():
                    print(f'{ntype} {k} {v.shape}', v[:,-1].max(), v[:,-1].mean())

        if echo: print(f'------\n')

    return g


# 标签特征传播
def label_feat_propagate(args, num_target_nodes, tgt_type, num_classes, init_labels, train_nid, val_nid, test_nid,init2sort, adjs):
    label_feats = {}
    # 特征传播设备为cpu
    prop_device = 'cpu'
    # 根据 args.label_feats 参数，决定是否进行标签传播
    if args.label_feats:
        # 如果数据集不是 IMDB，对训练集标签进行 one-hot 编码，构建 label_onehot。
        # 如果数据集是 IMDB，则直接使用初始化的标签，赋值给 label_onehot。
        if args.dataset != 'IMDB':
            label_onehot = torch.zeros((num_target_nodes, num_classes))
            # init_labels[train_nid] 作为输入，每个标签值（整数）会被转换为长度为 num_classes 的独热向量。例如，如果 num_classes 为 4，且 init_labels[train_nid] 中的某个标签为 2，则该标签将被转换为 [0, 0, 1, 0]。
            # 结果是一个大小为 (len(train_nid), num_classes) 的张量，表示训练集中所有节点的独热编码标签。
            label_onehot[train_nid] = F.one_hot(init_labels[train_nid], num_classes).float()
        else:
            label_onehot = torch.zeros((num_target_nodes, num_classes))
            label_onehot[train_nid] = init_labels[train_nid].float()

        # 根据不同数据集设置 extra_metapath（额外元路径）。在这里均为空列表。
        # 如果数据集不在上述列表中，则终止程序并抛出异常。
        if args.dataset == 'DBLP':
            extra_metapath = []
        elif args.dataset == 'IMDB':
            extra_metapath = []
        elif args.dataset == 'ACM':
            extra_metapath = []
        else:
            assert 0

        # 过滤 extra_metapath，保留路径长度大于 args.num_label_hops + 1 的路径。
        # max_length 是标签传播的最大路径长度，取 args.num_label_hops + 1 和 extra_metapath 中最长路径的最大值。
        extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_label_hops + 1]
        if len(extra_metapath):
            max_length = max(args.num_label_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = args.num_label_hops + 1
        # 打印标签传播的跳数，用于调试或检查。
        print(f'Current label-prop num hops = {args.num_label_hops}')

        # compute k-hop feature
        prop_tic = datetime.datetime.now()
        # 调用 hg_propagate_sparse_pyg 函数，基于指定的 args.num_label_hops 和 max_length 进行稀疏邻接矩阵传播，获取 meta_adjs。
        meta_adjs = hg_propagate_sparse_pyg(
            adjs, tgt_type, args.num_label_hops, max_length, extra_metapath, prop_feats=False, echo=False,
            prop_device=prop_device)
        # 打印每个元路径邻接矩阵的键名和尺寸信息。
        print(f'For label propagation, meta_adjs: (in SparseTensor mode)')
        for k, v in meta_adjs.items():
            print(k, v.sizes())
        print()

        # 遍历 meta_adjs，对每个邻接矩阵 v 进行处理，移除对角线元素 remove_diag(v)。
        # 乘以 label_onehot，将标签特征存入 label_feats。
        for k, v in tqdm(meta_adjs.items()):
            label_feats[k] = remove_diag(v) @ label_onehot
        # 手动触发垃圾回收，释放不必要的内存。
        gc.collect()

        # 检查标签特征的准确性，调用 check_acc 函数评估标签传播的结果。
        # 若数据集为 IMDB，设定损失类型为二元交叉熵 bce，否则采用默认损失。
        if args.dataset == 'IMDB':
            condition = lambda ra, rb, rc, k: True
            check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=False,
                      loss_type='bce')
        else:
            condition = lambda ra, rb, rc, k: True
            check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=True)
        print('Involved label keys', label_feats.keys())

        # 将 label_feats 中的每个特征根据 init2sort 重新排序，并保留符合 archs[args.dataset][1] 中的特征键。
        label_feats = {k: v[init2sort] for k, v in label_feats.items() if
                       k in archs[args.dataset][1]}  # if k in archs[args.dataset][1]

        prop_toc = datetime.datetime.now()
        print(f'Time used for label prop {prop_toc - prop_tic}')
    return label_feats


def hg_propagate_sparse_pyg(adjs, tgt_types, num_hops, max_length, extra_metapath, prop_feats=False, echo=False, prop_device='cpu'):
    store_device = 'cpu'
    if type(tgt_types) is not list:
        tgt_types = [tgt_types]

    label_feats = {k: v.clone() for k, v in adjs.items() if prop_feats or k[-1] in tgt_types} # metapath should start with target type in label propagation
    adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}

    for hop in range(2, max_length):
        reserve_heads = [ele[-(hop+1):] for ele in extra_metapath if len(ele) > hop]
        new_adjs = {}
        for rtype_r, adj_r in label_feats.items():
            metapath_types = list(rtype_r)
            if len(metapath_types) == hop:
                dtype_r, stype_r = metapath_types[0], metapath_types[-1]
                for rtype_l, adj_l in adjs_g.items():
                    dtype_l, stype_l = rtype_l
                    if stype_l == dtype_r:
                        name = f'{dtype_l}{rtype_r}'
                        if (hop == num_hops and dtype_l not in tgt_types and name not in reserve_heads) \
                          or (hop > num_hops and name not in reserve_heads):
                            continue
                        if name not in new_adjs:
                            if echo: print('Generating ...', name)
                            if prop_device == 'cpu':
                                new_adjs[name] = adj_l.matmul(adj_r)
                            else:
                                with torch.no_grad():
                                    new_adjs[name] = adj_l.matmul(adj_r.to(prop_device)).to(store_device)
                        else:
                            if echo: print(f'Warning: {name} already exists')
        label_feats.update(new_adjs)

        removes = []
        for k in label_feats.keys():
            metapath_types = list(k)
            if metapath_types[0] in tgt_types: continue  # metapath should end with target type in label propagation
            if len(metapath_types) <= hop:
                removes.append(k)
        for k in removes:
            label_feats.pop(k)
        if echo and len(removes): print('remove', removes)
        del new_adjs
        gc.collect()

    if prop_device != 'cpu':
        del adjs_g
        torch.cuda.empty_cache()

    return label_feats

def check_acc(preds_dict, condition, init_labels, train_nid, val_nid, test_nid, show_test=True, loss_type='ce'):
    mask_train, mask_val, mask_test = [], [], []
    remove_label_keys = []
    k = list(preds_dict.keys())[0]
    v = preds_dict[k]
    if loss_type == 'ce':
        na, nb, nc = len(train_nid), len(val_nid), len(test_nid)
    elif loss_type == 'bce':
        na, nb, nc = len(train_nid) * v.size(1), len(val_nid) * v.size(1), len(test_nid) * v.size(1)

    for k, v in preds_dict.items():
        if loss_type == 'ce':
            pred = v.argmax(1)
        elif loss_type == 'bce':
            pred = (v > 0).int()

        a, b, c = pred[train_nid] == init_labels[train_nid], \
                  pred[val_nid] == init_labels[val_nid], \
                  pred[test_nid] == init_labels[test_nid]
        ra, rb, rc = a.sum() / na, b.sum() / nb, c.sum() / nc

        if loss_type == 'ce':
            vv = torch.log(v / (v.sum(1, keepdim=True) + 1e-6) + 1e-6)
            la, lb, lc = F.nll_loss(vv[train_nid], init_labels[train_nid]), \
                         F.nll_loss(vv[val_nid], init_labels[val_nid]), \
                         F.nll_loss(vv[test_nid], init_labels[test_nid])
        else:
            vv = (v / 2. + 0.5).clamp(1e-6, 1-1e-6)
            la, lb, lc = F.binary_cross_entropy(vv[train_nid], init_labels[train_nid].float()), \
                         F.binary_cross_entropy(vv[val_nid], init_labels[val_nid].float()), \
                         F.binary_cross_entropy(vv[test_nid], init_labels[test_nid].float())
        if condition(ra, rb, rc, k):
            mask_train.append(a)
            mask_val.append(b)
            mask_test.append(c)
        else:
            remove_label_keys.append(k)
        # if show_test:
        #     print(k, ra, rb, rc, la, lb, lc, (ra/rb-1)*100, (ra/rc-1)*100, (1-la/lb)*100, (1-la/lc)*100)
        # else:
        #     print(k, ra, rb, la, lb, (ra/rb-1)*100, (1-la/lb)*100)
    print(set(list(preds_dict.keys())) - set(remove_label_keys))

    print((torch.stack(mask_train, dim=0).sum(0) > 0).sum() / na)
    print((torch.stack(mask_val, dim=0).sum(0) > 0).sum() / nb)
    if show_test:
        print((torch.stack(mask_test, dim=0).sum(0) > 0).sum() / nc)

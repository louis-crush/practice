import argparse

def parse_args_study(args=None):
    parser = argparse.ArgumentParser("CMS")
    parser.add_argument("--seeds", type=int, default=[1,2,3,4,5], help="the seed used in the training")
    parser.add_argument("-ld", "--log-dir", type=str, default="results", help="Dir for saving training results")
    parser.add_argument("--root", type=str, default="./data/")
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="DBLP")

    parser.add_argument("--num-hops", type=int, default=5, help="number of hops for propagation of raw labels")
    parser.add_argument("--label-feats", action='store_true', default=False, help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=2, help="number of hops for propagation of raw features")
    parser.add_argument("--ACM_keep_F", action='store_true', default=False, help="whether to use Field type")

    ## model
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--embed-size", type=int, default=512, help="inital embedding size of nodes with no attributes")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--input-drop", type=float, default=0.1,help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0., help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.,help="label feature dropout of model")

    ## network structure
    parser.add_argument("--n-layers-2", type=int, default=3,help="number of layers of the downstream task")
    parser.add_argument("--residual", action='store_true', default=False,help="whether to add residual branch the raw input features")
    parser.add_argument("--bns", action='store_true', default=False,help="whether to process the input features")

    ## training
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--amp", action='store_true', default=False, help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0)  # 1e-12

    # 论文未发表，暂时隐藏


    args = parser.parse_args()

    return args


def parse_args_search(args=None):
    parser = argparse.ArgumentParser("CMS")
    parser.add_argument("--seeds", type=int, default=[1], help="the seed used in the training")
    parser.add_argument("-ld", "--log-dir", type=str, default="results", help="Dir for saving training results")
    parser.add_argument("--root", type=str, default="./data/")
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="DBLP")

    parser.add_argument("--num-hops", type=int, default=5, help="number of hops for propagation of raw labels")
    parser.add_argument("--label-feats", action='store_true', default=False, help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=2, help="number of hops for propagation of raw features")
    parser.add_argument("--ACM_keep_F", action='store_true', default=False, help="whether to use Field type")

    ## model
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--embed-size", type=int, default=512, help="inital embedding size of nodes with no attributes")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--input-drop", type=float, default=0,help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0., help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.,help="label feature dropout of model")

    ## network structure
    parser.add_argument("--n-layers-2", type=int, default=3,help="number of layers of the downstream task")
    parser.add_argument("--residual", action='store_true', default=False,help="whether to add residual branch the raw input features")
    parser.add_argument("--bns", action='store_true', default=False,help="whether to process the input features")

    ## training
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--amp", action='store_true', default=False, help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0)  # 1e-12

    # 论文未发表，暂时隐藏


    args = parser.parse_args()

    return args
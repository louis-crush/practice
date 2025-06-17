# api.py

from fastapi import FastAPI, Query

from api_util import run_study, run_search
from params import parse_args_study, parse_args_search
from pydantic import BaseModel
from typing import Optional


app = FastAPI()

from types import SimpleNamespace

def parse_args_search_from_dict(params_dict):
    """
    将 dict 转换为仿 argparse.Namespace 的对象，用于替代 argparse
    """

    default_args = {
        "seeds": [1],
        "log_dir": "results",
        "root": "./data/",
        "cpu": False,
        "gpu": 0,
        "dataset": "DBLP",
        "num_hops": 5,
        "label_feats": False,
        "num_label_hops": 2,
        "ACM_keep_F": False,

        "num_epochs": 200,
        "embed_size": 512,
        "hidden_size": 512,
        "dropout": 0,
        "input_drop": 0,
        "att_drop": 0.,
        "label_drop": 0.,

        "n_layers_2": 3,
        "residual": False,
        "bns": False,

        "batch_size": 10000,
        "amp": False,
        "lr": 0.001,
        "weight_decay": 0,
        "eval_every": 1,
        "patience": 10,
        "eps": 0,

        "env_type": "node",
        "tau": 1,
        "lamda": 0.5,
        "env_layer_number": 2,
        "K": 3,

        "ns": 30,
        "alr": 0.001,
        "num_final": 60,
        "tau_max": 0.5,
        "tau_min": 0.01
    }

    default_args.update(params_dict)
    args = SimpleNamespace(**default_args)

    args.device = f"cuda:{args.gpu}" if not args.cpu else "cpu"

    # 返回类字典结构
    return args

@app.post("/search")
def search_path(
    dataset: str = Query("IMDB"),
    num_hops: int = Query(6),
    amp: bool = Query(True)
):
    args_dict = {
        "dataset": dataset,
        "num_hops": num_hops,
        "amp": amp
    }

    args = parse_args_search_from_dict(args_dict)

    best_meta, best_label = run_search(args)
    return {
        "best_meta_paths": best_meta,
        "best_label_paths": best_label
    }

from types import SimpleNamespace

def parse_args_study_from_dict(params_dict):
    """
    用于替代 argparse 的 dict 转换函数，供 FastAPI 使用
    """

    default_args = {
        "seeds": [1,2],
        "log_dir": "results",
        "root": "./data/",
        "cpu": False,
        "gpu": 0,
        "dataset": "DBLP",
        "num_hops": 5,
        "label_feats": False,
        "num_label_hops": 2,
        "ACM_keep_F": False,

        "num_epochs": 200,
        "embed_size": 512,
        "hidden_size": 512,
        "dropout": 0.45,
        "input_drop": 0.1,
        "att_drop": 0.,
        "label_drop": 0.,

        "n_layers_2": 3,
        "residual": False,
        "bns": False,

        "batch_size": 10000,
        "amp": False,
        "lr": 0.001,
        "weight_decay": 0.01,
        "eval_every": 1,
        "patience": 10,
        "eps": 0,

        "env_type": "node",
        "tau": 1,
        "lamda": 0.5,
        "env_layer_number": 2,
        "K": 3,

        "ns": 30,
        "alr": 0.001,
        "num_final": 60,
        "tau_max": 0.5,
        "tau_min": 0.01
    }

    default_args.update(params_dict)
    args = SimpleNamespace(**default_args)

    args.device = f"cuda:{args.gpu}" if not args.cpu else "cpu"

    # 返回类字典结构
    return args

@app.post("/study")
def study_path(
    dataset: str = Query("IMDB"),
    num_hops: int = Query(6),
    n_layers_2: int = Query(3),
    env_layer_number: int = Query(2),
    env_type: str = Query("node"),
    amp: bool = Query(True)
):
    args_dict = {
        "dataset": dataset,
        "num_hops": num_hops,
        "n_layers_2": n_layers_2,
        "env_layer_number": env_layer_number,
        "env_type": env_type,
        "amp": amp
    }

    args = parse_args_study_from_dict(args_dict)

    results = run_study(args)

    return results
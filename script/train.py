# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import time
import argparse

from datetime import timedelta
from typing import Optional, Any, Dict

from tool.utils.config import read_cfg, pretty_text
from tool.utils.dist_utils import init_dist, get_dist_info
from tool.utils.logging import get_logger
from tool.utils.env_collect import collect_env

from tool.trainer.train_sdk import train_api
from tool.trainer.utils import set_random_seed

from dataset import *
from modules.sparse4d_detector import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train E2E detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def main():
    args = parse_args()
    cfg = read_cfg(args.config)  # dict

    ## Save Train LogAndCfg
    if cfg.get("work_dir", None) is None:
        cfg["work_dir"] = os.path.join(
            "./e2e_worklog", os.path.splitext(os.path.basename(args.config))[0]
        )
    os.makedirs(cfg["work_dir"], exist_ok=True)
    pretty_cfg = pretty_text(cfg)
    with open(
        os.path.join(cfg["work_dir"], os.path.basename(args.config)),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(pretty_cfg)  # save *.py config file

    cfg["gpu_ids"] = (
        range(1) if cfg.get("gpu_ids", None) is None else range(cfg["gpu_ids"])
    )

    if args.resume_from is not None:
        cfg["resume_from"] = args.resume_from

    ## Create Logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg["work_dir"], f"{timestamp}.log")
    logger = get_logger(
        name="E2E", log_file=log_file, log_level=cfg.get("log_level", "INFO")
    )

    ## Start Pytoch Multiprocess
    if args.launcher == "none":
        distributed = False
    elif args.launcher == "pytorch":
        distributed = True
        # 使用spawn启动pytorch多进程和打开进程间的通信, timeout 60 mins.
        init_dist(args.launcher, timeout=timedelta(seconds=3600), **cfg["dist_params"])
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg["gpu_ids"] = range(world_size)
    else:
        raise NotImplementedError(f"not support {args.launcher} launcher.")

    ## Meta Infos
    meta = dict()
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    meta["env_info"] = env_info
    meta["config"] = pretty_cfg
    meta["exp_name"] = os.path.basename(args.config)
    meta["seed"] = cfg.get("seed", 0)

    dash_line = "\n" + "*" * 100 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    logger.info(f"Distributed training: {distributed}")
    # logger.info(f"Config:\n{pretty_cfg}")
    logger.info(
        f"Set random seed to {cfg.get('seed', 0)}, "
        f"deterministic: {args.deterministic}"
    )

    ## Set Seed
    set_random_seed(cfg.get("seed", 0), deterministic=args.deterministic)

    ## Build Model
    model = build_module(cfg["model"])
    model.init_weights()
    logger.info(f"Model:\n{model}")

    ## Build Dataset
    datasets = [build_module(cfg["data"]["train"])]
    logger.info(
        f"Load All TrainData Sum = [{len(datasets[0])}], Scene Sum = [{len(datasets[0]._scene)}]."
    )

    ## Call TrainApi
    train_api(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    args = parse_args()
    main()

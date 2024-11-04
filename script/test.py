# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import argparse

from typing import Optional, Dict, Any

from tool.trainer.utils import set_random_seed
from tool.utils.config import read_cfg
from tool.utils.dist_utils import init_dist, get_dist_info
from tool.utils.distributed import E2EDistributedDataParallel
from tool.utils.data_parallel import E2EDataParallel
from tool.runner.fp16_utils import wrap_fp16_model
from tool.runner.checkpoint import load_checkpoint

from dataset.dataloader_wrapper import *
from tool.tester.test_sdk import *

from dataset import *
from modules.sparse4d_detector import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train E2E detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--checkpoint", default="ckpt/sparse4dv3_r50.pth", help="checkpoint file"
    )
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        default="bbox",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox"',
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = read_cfg(args.config)  # dict
    cfg["model"]["img_backbone"]["init_cfg"] = {}

    ## Init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    ## Set Seed
    set_random_seed(cfg.get("seed", 0), deterministic=args.deterministic)

    ## Build Dataloader
    samples_per_gpu = cfg["data"]["test"].pop("samples_per_gpu", 1)
    dataset = build_module(cfg["data"]["test"])
    if distributed:
        data_loader = dataloader_wrapper(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg["data"]["workers_per_gpu"],
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )
    else:
        data_loader = dataloader_wrapper_without_dist(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg["data"]["workers_per_gpu"],
            dist=False,
            shuffle=False,
        )

    ## Build Model
    model = build_module(cfg["model"])
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    # print(checkpoint["meta"]["CLASSES"])

    ## GPU Inference
    if not distributed:
        model = E2EDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = E2EDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = custom_multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
        )

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {}
        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval"]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        if eval_kwargs.get("jsonfile_prefix", None) is None:
            import os

            eval_kwargs["jsonfile_prefix"] = os.path.join(
                "./e2e_worklog",
                os.path.splitext(os.path.basename(args.config))[0],
                "eval",
            )
        print("\n", eval_kwargs)
        results_dict = dataset.evaluate(outputs, **eval_kwargs)
        print(results_dict)


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


if __name__ == "__main__":
    args = parse_args()
    main()

# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import time
import torch

from typing import List, Dict, Optional, Any
from dataset import NuScenes4DDetTrackDataset
from dataset.dataloader_wrapper import dataloader_wrapper

from ..utils.logging import get_logger
from ..utils.distributed import E2EDistributedDataParallel
from ..utils.data_parallel import E2EDataParallel
from ..runner import build_optimizer, build_runner
from ..hook import *


def train_api(
    model,
    datasets: List,
    cfg: Dict,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
):
    logger = get_logger(name="E2E", log_file="INFO")

    ## Build Dataloader
    data_loaders = list()
    for dataset in datasets:
        data_loader = dataloader_wrapper(
            dataset,
            cfg["data"]["samples_per_gpu"],
            cfg["data"]["workers_per_gpu"],
            seed=cfg["seed"],
            runner_type=cfg["runner"]["type"],
            dist=distributed,
            num_gpus=len(cfg["gpu_ids"]),
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )
    data_loaders.append(data_loader)

    ## Build Parallel
    if distributed:
        model = E2EDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        model = E2EDataParallel(
            model.cuda(cfg["gpu_ids"][0]), device_ids=cfg["gpu_ids"]
        )

    ## Build Runner
    optimizer = build_optimizer(model, cfg["optimizer"])

    runner = build_runner(
        cfg["runner"],
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg["work_dir"],
            logger=logger,
            meta=meta,
        ),
    )

    runner.timestamp = timestamp

    ## Fp16 Optimizer Hooks Setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg["optimizer_config"], **fp16_cfg, distributed=distributed
        )
    # elif distributed and "type" not in cfg["optimizer_config"]:
    else:
        optimizer_config = OptimizerHook(**cfg["optimizer_config"])

    ## Register Hooks
    runner.register_training_hooks(
        cfg["lr_config"],
        optimizer_config,
        cfg["checkpoint_config"],
        cfg["log_config"],
    )

    # Register Eval Hooks
    if validate:
        val_samples_per_gpu = cfg["data"]["val"].pop("samples_per_gpu", 1)
        assert val_samples_per_gpu == 1
        val_dataset = build_module(cfg["data"]["val"])

        val_dataloader = dataloader_wrapper(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg["data"]["workers_per_gpu"],
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = False  # cfg.runner["type"] != "IterBasedRunner"
        eval_cfg["jsonfile_prefix"] = os.path.join(
            "val",
            cfg["work_dir"],
            time.ctime().replace(" ", "_").replace(":", "_"),
        )
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg["resume_from"]:
        runner.resume(cfg["resume_from"])
    elif cfg["load_from"]:
        runner.load_checkpoint(cfg["load_from"])

    runner.run(data_loaders, cfg["workflow"])


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)

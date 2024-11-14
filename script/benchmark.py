# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import time
import torch
import argparse

from tool.utils.config import read_cfg
from dataset.dataloader_wrapper import dataloader_wrapper
from typing import Optional, Any, Dict
from tool.runner.fp16_utils import wrap_fp16_model
from tool.runner.checkpoint import load_checkpoint

from tool.utils.data_parallel import E2EDataParallel
from modules.sparse4d_detector import Sparse4D
from dataset import NuScenes4DDetTrackDataset


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def parse_args():
    parser = argparse.ArgumentParser(description="E2E benchmark a model")
    parser.add_argument(
        "--config",
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
        help="test config file path",
    )
    parser.add_argument(
        "--checkpoint", default="ckpt/sparse4dv3_r50.pth", help="checkpoint file"
    )
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument("--samples", default=10, help="samples to benchmark")
    parser.add_argument("--log-interval", default=1, help="interval of logging")
    args = parser.parse_args()
    return args


def get_max_memory(model):
    device = getattr(model, "output_device", None)
    mem = torch.cuda.max_memory_allocated(device=device)
    mem_mb = torch.tensor([mem / (1024 * 1024)], dtype=torch.int, device=device)
    return mem_mb.item()


def main():
    args = parse_args()
    cfg = read_cfg(args.config)  # dict

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    # cfg["model"]["pretrained"] = None
    cfg["data"]["test"]["test_mode"] = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    samples_per_gpu = cfg["data"]["test"].pop("samples_per_gpu", 1)
    dataset = build_module(cfg["data"]["test"])
    data_loader = dataloader_wrapper(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg["data"]["workers_per_gpu"],
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    model = build_module(cfg["model"])
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = E2EDataParallel(model, device_ids=[0])

    model.eval()

    # # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    max_memory = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            start_time = time.perf_counter()
            model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            max_memory = max(max_memory, get_max_memory(model))

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f"Done image [{i + 1:<3}/ {args.samples}], "
                    f"fps: {fps:.1f} img / s, "
                    f"gpu mem: {max_memory} M"
                )

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f"Overall fps: {fps:.1f} img / s")
            break


if __name__ == "__main__":
    main()

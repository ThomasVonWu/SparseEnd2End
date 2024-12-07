# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
"""
Generate onboard config parameters  file  and model input bin file offline.
    1) kmeans_anchor_900x11_float32.bin
    2) lidar2img6x4x4.bin (each frame's lidar2img is different)
"""

import yaml
import numpy as np

from tqdm import tqdm
from typing import Optional, Any, Dict

from tool.utils.config import read_cfg
from tool.trainer.utils import set_random_seed
from tool.runner.checkpoint import load_checkpoint

from dataset.nuscenes_dataset import *
from dataset.dataloader_wrapper.dataloader_wrapper import dataloader_wrapper
from dataset.utils.scatter_gather import scatter

from modules.sparse4d_detector import Sparse4D


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def gen_kmeans_anchor(model_path, model_cfg):

    model = build_module(model_cfg)
    load_checkpoint(model, model_path, map_location="cpu")

    anchor = model.head.instance_bank.anchor.data.numpy()
    anchor_shape = "*".join([str(it) for it in anchor.shape])
    path = f"onboard/assets/instance_bank_anchor_{anchor_shape}_{anchor.dtype}.bin"
    anchor.tofile(path)
    print(f"Save instance bank anchor as bin file in: {path}.")


def gen_lidar2img_calib_params(config, seed=100, num_frames=5):
    """
    Eeach frame's lidar2img calibration parameter  is different, so, we must save each frame's  parameter.
    """
    dataset_type = config.copy()["data"]["val"].pop("type")
    dataset = eval(dataset_type)(**cfg["data"]["val"])

    dataloader = dataloader_wrapper(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        seed=seed,
        dist=False,
        shuffle=False,
    )
    data_iter = dataloader.__iter__()

    lidar2imgs = []
    for _ in tqdm(range(num_frames)):
        data = next(data_iter)
        data = scatter(data, [0])[0]
        lidar2img = data["lidar2img"][0].cpu().numpy()
        lidar2imgs.append(lidar2img)

    lidar2imgs_npy = np.stack(lidar2imgs, axis=0)
    lidar2imgs_shape = "*".join([str(it) for it in lidar2imgs_npy.shape])
    path = f"onboard/assets/lidar2img_{lidar2imgs_shape}_{lidar2imgs_npy.dtype}.bin"
    lidar2imgs_npy.tofile(path)
    print(f"Save lidar2img calibration params as bin file in: {path}.")


if __name__ == "__main__":

    config_path = "dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py"
    model_path = "ckpt/sparse4dv3_r50.pth"
    cfg = read_cfg(config_path)
    set_random_seed(seed=100, deterministic=True)

    gen_lidar2img_calib_params(cfg, seed=cfg["seed"])
    gen_kmeans_anchor(model_path, cfg["model"])

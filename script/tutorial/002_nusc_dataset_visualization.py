# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import cv2
import numpy as np

from typing import Dict
from tool.utils.config import read_cfg
from dataset.nuscenes_dataset import *
from tool.trainer.utils import set_random_seed
from dataset.utils.scatter_gather import scatter
from tool.visualization.utils import draw_lidar_bbox3d_metas
from dataset.dataloader_wrapper.dataloader_wrapper import dataloader_wrapper


def val_pipeline_vis(
    config: Dict,
    offline=False,
    save_dir="data/nusc_anno_vis/val",
    seed=100,
    show_lidarpts=False,
):

    dataset_type = cfg.copy()["data"]["val"].pop("type")
    dataset = eval(dataset_type)(**cfg["data"]["val"])

    dataloader = dataloader_wrapper(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        seed=seed,
        dist=False,
        shuffle=False,
    )
    data_iter = dataloader.__iter__()
    img_norm_mean = np.array(config["img_norm_cfg"]["mean"])
    img_norm_std = np.array(config["img_norm_cfg"]["std"])
    i = 0
    for _ in range(50):
        data = next(data_iter)
        data = scatter(data, [0])[0]
        raw_imgs = data["img"][0].permute(0, 2, 3, 1).cpu().numpy()
        raw_imgs = (
            raw_imgs * img_norm_std + img_norm_mean
        )  # float64/(6, 256, 704, 3)/rgb

        anchor = data["gt_bboxes_3d"][0]
        label = data["gt_labels_3d"][0]
        if not show_lidarpts:
            img_show = draw_lidar_bbox3d_metas(
                anchor,
                label,
                raw_imgs,
                data["lidar2img"][0],
                data["img_metas"][0],
            )
        else:
            img_show = draw_lidar_bbox3d_metas(
                anchor,
                label,
                raw_imgs,
                data["lidar2img"][0],
                data["img_metas"][0],
                # data["gt_depth"][0], # method1
                data["gt_depth_ori"][0],
                show_lidarpts=True,
            )
        img_show = img_show[..., ::-1]  # rgb to bgr
        cv2.imshow("val_pipeline", img_show)
        cv2.waitKey(0)
        if offline:
            sample_idx = data["img_metas"][0]["sample_idx"]
            sample_scene = data["img_metas"][0]["sample_scene"]
            save_dirs = f"{save_dir}/{sample_scene}"
            os.makedirs(save_dirs, exist_ok=True)
            cv2.imwrite(f"{save_dirs}/{i}_{sample_idx}.jpg", img_show)
            i += 1


def train_pipeline_vis(
    config: Dict,
    offline=False,
    save_dir="data/nusc_anno_vis/train",
    seed=100,
    show_lidarpts=False,
):

    dataset_type = cfg.copy()["data"]["train"].pop("type")
    dataset = eval(dataset_type)(**cfg["data"]["train"])

    dataloader = dataloader_wrapper(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        seed=seed,
        runner_type="IterBasedRunner",
        dist=False,
        shuffle=False,
    )

    data_iter = dataloader.__iter__()
    img_norm_mean = np.array(config["img_norm_cfg"]["mean"])
    img_norm_std = np.array(config["img_norm_cfg"]["std"])
    i = 0
    for _ in range(50):
        data = next(data_iter)
        data = scatter(data, [0])[0]
        raw_imgs = data["img"][0].permute(0, 2, 3, 1).cpu().numpy()
        raw_imgs = (
            raw_imgs * img_norm_std + img_norm_mean
        )  # float64/(6, 256, 704, 3)/rgb

        anchor = data["gt_bboxes_3d"][0]
        label = data["gt_labels_3d"][0]

        if not show_lidarpts:
            img_show = draw_lidar_bbox3d_metas(
                anchor,
                label,
                raw_imgs,
                data["lidar2img"][0],
                data["img_metas"][0],
            )
        else:
            img_show = draw_lidar_bbox3d_metas(
                anchor,
                label,
                raw_imgs,
                data["lidar2img"][0],
                data["img_metas"][0],
                # data["gt_depth"][0], # method1
                data["gt_depth_ori"][0],
                show_lidarpts=True,
            )
        img_show = img_show[..., ::-1]  # rgb to bgr
        cv2.imshow("pipeline", img_show)
        cv2.waitKey(0)
        if offline:
            sample_idx = data["img_metas"][0]["sample_idx"]
            sample_scene = data["img_metas"][0]["sample_scene"]
            save_dirs = f"{save_dir}/{sample_scene}"
            os.makedirs(save_dirs, exist_ok=True)
            cv2.imwrite(f"{save_dirs}/{i}_{sample_idx}.jpg", img_show)
            i += 1


if __name__ == "__main__":
    config_path = "dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py"
    cfg = read_cfg(config_path)
    set_random_seed(seed=cfg["seed"], deterministic=True)
    # val_pipeline_vis(cfg, offline=True, seed=cfg["seed"], show_lidarpts=True)
    train_pipeline_vis(cfg, offline=True, seed=cfg["seed"], show_lidarpts=False)

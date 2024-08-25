# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

import os
import numpy as np
from typing import List


def save_bin_format(x, x_name: str, save_prefix, sample_index, logger):
    """x : numpy.ndarray"""
    logger.debug(
        f"{x_name}\t:\t{x.flatten()[:5]} ... {x.flatten()[-5:]}, min={x.min()}, max={x.max()}"
    )

    x_shape = "x".join([str(it) for it in x.shape])
    x_path = os.path.join(
        save_prefix,
        f"sample_{sample_index}_{x_name}_{x_shape}_{x.dtype}.bin",
    )
    x.tofile(x_path)
    logger.info(f"Save data bin file: {x_path}.")


def save_bins_backbone(
    img: np.ndarray,
    feature: np.ndarray,
    sample_index: int,
    logger,
    save_prefix: str = "script/tutorial_task_smart/asset",
):
    os.makedirs(save_prefix, exist_ok=True)

    for x, x_name in zip(
        [img, feature],
        ["img", "feature"],
    ):
        save_bin_format(x, x_name, save_prefix, sample_index, logger)


def save_bins_1stframe_head(
    inputs: List[np.ndarray],
    outputs: List[np.ndarray],
    sample_index: int,
    logger,
    save_prefix: str = "script/tutorial_task_smart/asset",
):
    os.makedirs(save_prefix, exist_ok=True)
    assert len(inputs) == 11
    assert len(outputs) == 4

    for x, x_name in zip(
        inputs + outputs,
        [
            "instance_feature",
            "anchor",
            "time_interval",
            "feature",
            "spatial_shapes",
            "level_start_index",
            "image_wh",
            "lidar2cam",
            "cam_distortion",
            "cam_intrinsic",
            "aug_mat",
            "pred_instance_feature",
            "pred_anchor",
            "pred_class_score",
            "pred_quality",
        ],
    ):
        save_bin_format(x, x_name, save_prefix, sample_index, logger)


def save_bins_head(
    inputs: List[np.ndarray],
    outputs: List[np.ndarray],
    sample_index: int,
    logger,
    save_prefix: str = "script/tutorial_task_smart/asset",
):
    os.makedirs(save_prefix, exist_ok=True)
    assert len(inputs) == 15
    assert len(outputs) == 5

    for x, x_name in zip(
        inputs + outputs,
        [
            "temp_instance_feature",
            "temp_anchor",
            "mask",
            "track_id",
            "instance_feature",
            "anchor",
            "time_interval",
            "feature",
            "spatial_shapes",
            "level_start_index",
            "image_wh",
            "lidar2cam",
            "cam_distortion",
            "cam_intrinsic",
            "aug_mat",
            "pred_instance_feature",
            "pred_anchor",
            "pred_class_score",
            "pred_quality",
            "pred_track_id",
        ],
    ):
        save_bin_format(x, x_name, save_prefix, sample_index, logger)


def save_bins(
    inputs: List[np.ndarray],
    outputs: List[np.ndarray],
    names: str,
    sample_index: int,
    logger,
    save_prefix: str = "script/tutorial_task_smart/asset",
):
    os.makedirs(save_prefix, exist_ok=True)

    for x, x_name in zip(inputs + outputs, names):
        save_bin_format(x, x_name, save_prefix, sample_index, logger)

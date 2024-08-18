# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import torch.nn as nn

from typing import Union, Dict
from .base_loss import *


from dataset.config.nusc_std_bbox3d import *


class SparseBox3DLoss(nn.Module):
    def __init__(
        self,
        loss_box: Union[Dict, None],
        loss_centerness: Union[Dict, None],
        loss_yawness: Union[Dict, None],
        cls_allow_reverse=None,
    ):
        super().__init__()

        # =========== build modules ===========
        def build_module(cfg):
            cfg2 = cfg.copy()
            type = cfg2.pop("type")
            return eval(type)(**cfg2)

        self.loss_box = build_module(loss_box)
        self.loss_cns = build_module(loss_centerness)
        self.loss_yns = build_module(loss_yawness)
        self.cls_allow_reverse = cls_allow_reverse

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        # Some categories do not distinguish between positive and negative
        # directions. For example, barrier in nuScenes dataset.
        if self.cls_allow_reverse is not None and cls_target is not None:
            if_reverse = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                < 0
            )
            if_reverse = (
                torch.isin(cls_target, cls_target.new_tensor(self.cls_allow_reverse))
                & if_reverse
            )
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],
                box_target[..., [SIN_YAW, COS_YAW]],
            )

        output = {}
        box_loss = self.loss_box(box, box_target, weight=weight, avg_factor=avg_factor)
        output[f"loss_box{suffix}"] = box_loss

        if quality is not None:
            cns = quality[..., CNS]
            yns = quality[..., YNS].sigmoid()
            cns_target = torch.norm(
                box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1
            )
            cns_target = torch.exp(-cns_target)
            cns_loss = self.loss_cns(cns, cns_target, avg_factor=avg_factor)
            output[f"loss_cns{suffix}"] = cns_loss

            yns_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                > 0
            )
            yns_target = yns_target.float()
            yns_loss = self.loss_yns(yns, yns_target, avg_factor=avg_factor)
            output[f"loss_yns{suffix}"] = yns_loss
        return output

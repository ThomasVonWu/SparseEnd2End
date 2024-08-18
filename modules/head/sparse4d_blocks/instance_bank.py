# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from .sparse3d_embedding import *

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    """
    confidence  : torch.tensor, shape(bs, num_querys)
    inputs:
        instance_feature : torch.tensor, shape(bs, num_querys, 256)
        anchor  : torch.tensor, shape(bs, num_querys, 10)
        cls     : torch.tensor, shape(bs, num_querys, 11)
    """
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)  # (bs, k), (bs, k)
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * N).reshape(
        -1
    )  # (bs, k) + (k, 1) => (bs, k) => (bs*k,)
    outputs = []
    # (bs, num_querys, c) => (bs, k, 256)
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        # =========== build modules ===========
        def build_module(cfg):
            cfg2 = cfg.copy()
            type = cfg2.pop("type")
            return eval(type)(**cfg2)

        if anchor_handler is not None:
            anchor_handler = build_module(anchor_handler)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.track_id = None
        self.prev_id = 0

    def get(self, batch_size, metas=None, dn_metas=None):
        """
        Return:
            instance_feature : Tensor.shape(bs, 900, 25)
            anchor : Tensor.shape(bs, 900, 11)
            self.cached_feature: None or
            self.cached_anchor: None or
            time_interval: TensorShape (bs, )
        """
        instance_feature = self.instance_feature[None].repeat((batch_size, 1, 1))
        anchor = self.anchor[None].repeat((batch_size, 1, 1))

        if self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval  # < 2s

            if self.anchor_handler is not None:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["global2lidar"]
                            @ self.metas["img_metas"][i]["lidar2global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )  # (1, 4, 4)
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):  # train mode step in
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        """ "
        Args:
            instance_feature: TensorShape(bs, 1220, 256)
            anchor: TensorShape(bs, 1220, 11)
        Return:

        """
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=1)
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=1)
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        if self.track_id is not None:
            self.track_id = torch.where(
                self.mask[:, None],
                self.track_id,
                self.track_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()  # (B, num_querys)
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def get_track_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()  # (bs, num_querys)
        track_id = confidence.new_full(confidence.shape, -1).long()

        if self.track_id is not None and self.track_id.shape[0] == track_id.shape[0]:
            track_id[:, : self.track_id.shape[1]] = self.track_id

        mask = track_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(track_id) + self.prev_id
        track_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        self.update_track_id(track_id, confidence)
        return track_id

    def update_track_id(self, track_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        track_id = topk(temp_conf, self.num_temp_instances, track_id)[1][0]
        track_id = track_id.squeeze(dim=-1)  # (bs, k)
        self.track_id = F.pad(
            track_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )  # (bs, num_querys)

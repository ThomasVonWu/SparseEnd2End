# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from typing import List, Optional
from .sparse3d_embedding import *
from torch.cuda.amp.autocast_mode import autocast
from modules.cnn.base_module import Sequential, BaseModule
from modules.cnn.module import xavier_init, constant_init

try:
    from ...ops import deformable_aggregation_function as DAF
except:
    DAF = None

__all__ = [
    "DeformableAttentionAggr",
    "DenseDepthNet",
    "AsymmetricFFN",
    "GridMask",
]


class DeformableAttentionAggr(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableAttentionAggr, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims

        # =========== build modules ===========
        def build_module(cfg):
            cfg2 = cfg.copy()
            type = cfg2.pop("type")
            return eval(type)(**cfg2)

        self.kps_generator = build_module(kps_generator)
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_module(temporal_fusion_module)
        else:
            self.temp_module = None
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(*linear_relu_ln(embed_dims, 1, 2, 12))
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        """
        Args:
            instance_feature: (bs, 900+x, 256) train:x=5*32*2 test=0;
            anchor: (bs, 900+x, 11) train:x=5*32*2 test=0;
            anchor_embed: (bs, 900+x, 256) train:x=5*32*2 test=0;
        Return:
            out: (bs, 900+x, 512)
        """
        bs, num_anchor = instance_feature.shape[:2]
        # [bs, 1220, 7+6, 3]
        key_points = self.kps_generator(anchor, instance_feature)
        # [bs, 1220, 6, 4, 13, 8]
        weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:
            points_2d = (
                self.project_points(
                    key_points,
                    metas["lidar2img"],  # lidar2img
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,  # 1220
                    self.num_pts,  # 13
                    self.num_cams,  # 6
                    self.num_levels,  # 4
                    self.num_groups,  # 8
                )
            )
            # (bs, 1220, 256)
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["lidar2img"],
                metas.get("image_wh"),
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(
                dim=2
            )  # fuse multi-point features =>(bs, 1220, 256)
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        """
        instance_feature: (bs, 900+x, 256) train:x=5*32*2 test=0;
        anchor_embed: (bs, 900+x, 256) train:x=5*32*2 test=0;

        self.camera_encoder:
            Sequential(
                (0): Linear(in_features=12, out_features=256, bias=True)
                (1): ReLU(inplace=True)
                (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                (3): Linear(in_features=256, out_features=256, bias=True)
                (4): ReLU(inplace=True)
                (5): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                )
        self.weights_fc:
            Linear(in_features=256, out_features=416, bias=True)
        """
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            # 对相机的外参编码： (bs,6,4,4)=>(bs,6,256)
            camera_embed = self.camera_encoder(
                metas["lidar2img"][:, :, :3].reshape(bs, self.num_cams, -1)
            )
            # [bs, 900+x, 1, 256] + [bs, 1, 6, 256] => (bs, 900+x, 6, 256)
            feature = feature[:, :, None] + camera_embed[:, None]

        # [] => [1, 1220, 6, 4, 13, 8]
        weights = (
            self.weights_fc(feature)  # (bs, 900+x, 6, 256)=>(bs, 900+x, 6, 416)
            .reshape(bs, num_anchor, -1, self.num_groups)  # =>(bs, 900+x, 312, 8)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        # self.attn_drop=0.15
        if self.training and self.attn_drop > 0:
            # [1, 1220, 6, 1, 13, 1]
            mask = torch.rand(bs, num_anchor, self.num_cams, 1, self.num_pts, 1)
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)
        return weights

    @staticmethod
    def project_points(key_points, lidar2img, image_wh=None):
        """
        Args:
            key_points : Shape[1, 1220, 13, 3].
        Return:
            key_points2d : Shape[1, 6, 1220, 13, 2].


        """
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        # points_2d = torch.matmul(
        #     lidar2img[:, :, None, None], pts_extend[:, None, ..., None]
        # ).squeeze(-1)
        points_2d = torch.matmul(
            lidar2img[:, :, None, None], pts_extend[:, None, ..., None]
        )[..., 0]
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        lidar2img: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feature_maps: List[Tensor]
                        (bs, 6, 256, 64, 176) float32
                        (bs, 6, 256, 32, 88)  float32
                        (bs, 6, 256, 16, 44)  float32
                        (bs, 6, 256, 8,  22)  float32
            key_points: TensorShape=(bs, 1220, 12, 3)
        Returns:
            TensorShape=(bs, num_anchor, num_cams, num_levels, num_pts, embed_dims)
        """
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableAttentionAggr.project_points(
            key_points, lidar2img, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(
            end_dim=1
        )  # (bs, 6, 1220, 13, 2)=> (bs*6, 1220, 13, 2)

        features = []
        # (6*bs, 256, 1220, 12)
        # (6*bs, 256, 1220, 12)
        # (6*bs, 256, 1220, 12)
        # (6*bs, 256, 1220, 12)
        for fm in feature_maps:
            features.append(nn.functional.grid_sample(fm.flatten(end_dim=1), points_2d))
        features = torch.stack(features, dim=1)  # (6*bs, 4, 256, 1220, 12)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        """
        Args:
            features: TensorShape(bs, num_anchor, num_cams, num_levels, num_pts, embed_dims)
                                 (bs, 1220, num_cams, 4, 13, 256)
            weights: TensorShape(bs, 1220, num_cams, 4, 13, 8)
        Return:
            features: (bs, num_anchor, self.num_pts, self.embed_dims)
                      (1,1220,13,256)
        """
        bs, num_anchor = weights.shape[:2]
        # (bs, 1220, 6, 4, 13, 8, 1) * (bs, 1220, 6, 4, 13, 8, 32)  => (bs, 1220, 6, 4, 13, 8, 32)
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        # 将num_cams和num_levels维度合并：(1, 1220, 13, 8, 32)
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims)
        return features


class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(gt > 0.0, torch.logical_not(torch.isnan(pred)))
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = error / max(1.0, len(gt) * len(depth_preds)) * self.loss_weight
            loss = loss + _loss
        return loss


class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=256 * 2,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = nn.ReLU(inplace=True)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        self.pre_norm = nn.modules.normalization.LayerNorm(normalized_shape=512)

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = nn.Identity()
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                nn.Identity()
                if in_channels == embed_dims
                else nn.Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)


class Grid(object):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img, label


class GridMask(nn.Module):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = torch.from_numpy(mask.copy()).float().cuda()
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float().cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)

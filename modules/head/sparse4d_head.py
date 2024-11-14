# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import torch.nn as nn

from typing import List, Optional, Union
from torch.nn.modules.normalization import *

from modules.cnn.base_module import BaseModule
from tool.utils.dist_utils import reduce_mean
from modules.cnn.transformer import *

from .sparse4d_blocks.instance_bank import *
from .sparse4d_blocks.core_blocks import *
from .sparse4d_blocks.sparse3d_embedding import *
from .target import *
from .decoder import *
from .loss.base_loss import *
from .loss.sparse4d_losses import *

from tool.runner.fp16_utils import force_fp32

__all__ = ["Sparse4DHead"]


class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========
        def build_module(cfg):
            cfg2 = cfg.copy()
            type = cfg2.pop("type")
            return eval(type)(**cfg2)

        self.instance_bank = build_module(instance_bank)
        self.anchor_encoder = build_module(anchor_encoder)
        self.sampler = build_module(sampler)
        self.decoder = build_module(decoder)
        self.loss_cls = build_module(loss_cls)
        self.loss_reg = build_module(loss_reg)
        self.op_config_map = {
            "temp_gnn": temp_graph_model,
            "gnn": graph_model,
            "norm": norm_layer,
            "ffn": ffn,
            "deformable": deformable_model,
            "refine": refine_layer,
        }
        self.layers = nn.ModuleList(
            [build_module(self.op_config_map.get(op)) for op in self.operation_order]
        )
        self.embed_dims = self.instance_bank.embed_dims
        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)  # (1, 1220, 512)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)  # (1, 1220, 256) => (1, 1220, 512)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):

        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        (
            instance_feature,  # (1, 900, 256) float32
            anchor,  # (1, 900, 11) float32
            temp_instance_feature,  # None
            temp_anchor,  # None
            time_interval,  # (1,)=0.5000 float32
        ) = self.instance_bank.get(batch_size, metas, dn_metas=self.sampler.dn_metas)

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if "track_id" in metas["img_metas"][0]:

                gt_track_id = [
                    torch.from_numpy(x["track_id"]).cuda() for x in metas["img_metas"]
                ]  # [(nums_gt,), ...]
            else:
                gt_track_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_track_id,
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[
                1
            ]  # num_dn_groups*num_gt=5*32*2(neg_noise->2)
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)  # (bs, 320+900, 11)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )  # (bs, 320+900, 256)
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor  # 320+900-320=900
            attn_mask = anchor.new_ones((num_instance, num_instance), dtype=torch.bool)
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[
                num_free_instance:, num_free_instance:
            ] = dn_attn_mask  # (1120, 1120)

        anchor_embed = self.anchor_encoder(anchor)  # (bs, 320+900, 256)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # =================== forward the layers ====================
        prediction = []  # output=6 ([1, 900, 11] * 6 ) float32
        classification = (
            []
        )  # output=6 ([1, 900, 10], None, None, None, None, None, [1, 900, 10]) float32
        quality = (
            []
        )  # output=6 ([1, 900, 2], None, None, None, None, None, [1, 900, 2]) float32
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask if temp_instance_feature is None else None,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":  # [1, 900, 512] => [1, 900, 256]
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":  # [1, 900, 256]
                # i = 0, 7
                instance_feature = self.layers[i](
                    instance_feature,  # [1, 900, 256]
                    anchor,  # [1, 900, 11]
                    anchor_embed,  # [1, 900, 256]
                    feature_maps,  # [[1, 89760, 256], [6, 4, 2], [6, 4, 4]]
                    metas,
                )
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0  # default=3
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,  # None
                        )
                if i != len(self.operation_order) - 1:
                    # (1, 1220, 11) => (1, 1220, 256)
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:

            dn_classification = [x[:, num_free_instance:] for x in classification]
            classification = [
                x[:, :num_free_instance] for x in classification
            ]  # [(1, 900, 10), ...,] 6
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None for x in quality
            ]
            # 1) split noisy instance
            output.update(
                {
                    "dn_prediction": dn_prediction,  # [(1, 320, 11),...] 6
                    "dn_classification": dn_classification,  # [(1, 320, 10), ...,] 6
                    "dn_reg_target": dn_reg_target,  # (1, 320, 10)
                    "dn_cls_target": dn_cls_target,  # (1, 320)
                    "dn_valid_mask": valid_mask,  # (1, 320)
                }
            )

            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,  # (1, 320, 10)
                        "temp_dn_cls_target": temp_dn_cls_target,  # (1, 320)
                        "temp_dn_valid_mask": temp_valid_mask,  # (1, 320)
                        "dn_id_target": dn_id_target,  # (1, 320)
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            # cache dn_metas for temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        # 2) split learnable instance
        output.update(
            {
                "classification": classification,  # list:length=6 ([1, 900, 10], None, None, None, None, [1, 900, 10])
                "prediction": prediction,  # list:length=6 ([1, 900, 11], ..., [1, 900, 11])
                "quality": quality,  # list:length=6 ([1, 900, 2], ..., [1, 900, 2])
            }
        )

        # cache current instances for temporal modeling
        # input: [1, 900, 256], [1, 900, 11], [1, 900, 10],  metas, feature_maps
        self.instance_bank.cache(instance_feature, anchor, cls, metas, feature_maps)
        if not self.training:
            track_id = self.instance_bank.get_track_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["track_id"] = track_id  # [1, 900], int64
        return output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if "dn_prediction" not in model_outs:
            return output

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(zip(dn_cls_scores, dn_reg_preds)):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(end_dim=1)[
            dn_valid_mask
        ]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(end_dim=1)[
            dn_valid_mask
        ][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("track_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )

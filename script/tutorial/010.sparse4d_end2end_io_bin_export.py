# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import argparse
import torch
import torch.nn as nn
import logging
from inspect import signature
from typing import Union, Optional, Any, Dict, List

from tool.utils.config import read_cfg
from tool.utils.logger import set_logger

# from tool.runner.fp16_utils import auto_fp16
from tool.runner.fp16_utils import wrap_fp16_model
from tool.runner.checkpoint import load_checkpoint
from tool.trainer.utils import set_random_seed
from tool.utils.save_bin import save_bins

from modules.sparse4d_detector import Sparse4D
from dataset.dataloader_wrapper import dataloader_wrapper
from dataset import NuScenes4DDetTrackDataset
from dataset.utils.scatter_gather import scatter


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def parse_args():
    parser = argparse.ArgumentParser(description="E2E inference with bin export!")
    parser.add_argument(
        "--config",
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
        help="inference config file path",
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
    parser.add_argument(
        "--log",
        type=str,
        default="script/tutorial/save_bin.log",
    )
    args = parser.parse_args()
    return args


class Sparse4D_backbone(nn.Module):
    def __init__(self, model):
        super(Sparse4D_backbone, self).__init__()
        self._model = model

    def io_hook(self):
        """IO tensor is converted to numpy type."""
        img = self._img.detach().cpu().numpy()
        feature = self._feature.detach().cpu().numpy()
        return img, feature

    def feature_maps_format(self, feature_maps):

        bs, num_cams = feature_maps[0].shape[:2]
        spatial_shape = []

        col_feats = []  # (bs, n, c, -1)
        for i, feat in enumerate(feature_maps):
            spatial_shape.append(feat.shape[-2:])
            col_feats.append(torch.reshape(feat, (bs, num_cams, feat.shape[2], -1)))

        # (bs, n, c', c) => (bs, n*c', c)
        col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
        spatial_shape = [spatial_shape] * num_cams
        spatial_shape = torch.tensor(
            spatial_shape,
            dtype=torch.int64,
            device=col_feats.device,
        )

        scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
        scale_start_index = scale_start_index.flatten().cumsum(dim=0)
        scale_start_index = torch.cat(
            [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
        )
        scale_start_index = scale_start_index.reshape(num_cams, -1)

        feature_maps = [
            col_feats,
            spatial_shape,
            scale_start_index,
        ]
        return feature_maps

    # @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        self._img = img.clone()
        if img.dim() == 5:
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self._model.use_grid_mask:
            img = self._model.grid_mask(img)
        if "metas" in signature(self._model.img_backbone.forward).parameters:
            feature_maps = self._model.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self._model.img_backbone(img)
        if self._model.img_neck is not None:
            feature_maps = list(self._model.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(feat, (bs, num_cams) + feat.shape[1:])
        if return_depth and self._model.depth_branch is not None:
            depths = self._model.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self._model.use_deformable_func:
            feature_maps = self.feature_maps_format(feature_maps)

        self._feature = (feature_maps[0]).clone()

        if return_depth:
            return feature_maps, depths
        return feature_maps

    def forward(self, img):
        return self.extract_feat(img)


class Sparse4D_head(nn.Module):
    def __init__(self, model):
        super(Sparse4D_head, self).__init__()
        self.head = model
        self.first_frame = True

    def io_hook(
        self,
    ):
        first_frame = True
        if (
            self._temp_instance_feature is not None
            and self._temp_anchor is not None
            and self._mask is not None
            and self._track_id is not None
            and self._pred_track_id is not None
        ):
            first_frame = False
            temp_instance_feature = self._temp_instance_feature.detach().cpu().numpy()
            temp_anchor = self._temp_anchor.detach().cpu().numpy()
            mask = self._mask.int().detach().cpu().numpy()
            track_id = self._track_id.int().detach().cpu().numpy()
            pred_track_id = self._pred_track_id.int().detach().cpu().numpy()

        instance_feature = self._instance_feature.detach().cpu().numpy()
        anchor = self._anchor.detach().cpu().numpy()
        time_interval = self._time_interval.detach().cpu().numpy()
        feature = self._feature.detach().cpu().numpy()
        spatial_shapes = self._spatial_shapes.int().detach().cpu().numpy()
        level_start_index = self._level_start_index.int().detach().cpu().numpy()
        image_wh = self._image_wh.detach().cpu().numpy()
        lidar2cam = self._lidar2cam.detach().cpu().numpy()
        cam_distortion = self._cam_distortion.detach().cpu().numpy()
        cam_intrinsic = self._cam_intrinsic.detach().cpu().numpy()
        aug_mat = self._aug_mat.detach().cpu().numpy()

        pred_instance_feature = self._pred_instance_feature.detach().cpu().numpy()
        pred_anchor = self._pred_anchor.detach().cpu().numpy()
        pred_class_score = self._pred_class_score.detach().cpu().numpy()
        pred_quality = self._pred_quality.detach().cpu().numpy()

        if first_frame:
            # print("instance_feature:\n", instance_feature.flatten()[:6])
            # print("anchor:\n", anchor.flatten()[:6])
            # print("time_interval:\n", time_interval.flatten()[:6])
            # print("feature:\n", feature.flatten()[:6])
            # print("spatial_shapes:\n", spatial_shapes.flatten()[:6])
            # print("level_start_index:\n", level_start_index.flatten()[:6])
            # print("image_wh:\n", image_wh.flatten()[:6])
            # print("lidar2cam:\n", lidar2cam.flatten()[:6])
            # print("cam_distortion:\n", cam_distortion.flatten()[:6])
            # print("cam_intrinsic:\n", cam_intrinsic.flatten()[:6])
            # print("aug_mat:\n", aug_mat.flatten()[:6])
            inputs = [
                instance_feature,
                anchor,
                time_interval,
                feature,
                spatial_shapes,
                level_start_index,
                image_wh,
                lidar2cam,
                cam_distortion,
                cam_intrinsic,
                aug_mat,
            ]
            # print("pred_instance_feature:\n", pred_instance_feature.flatten()[:5])
            # print("pred_anchor:\n", pred_anchor.flatten()[:5])
            # print("pred_class_score:\n", pred_class_score.flatten()[:5])
            # print("pred_quality:\n", pred_quality.flatten()[:5])
            outputs = [
                pred_instance_feature,
                pred_anchor,
                pred_class_score,
                pred_quality,
            ]
            return inputs, outputs, first_frame
        inputs = [
            temp_instance_feature,
            temp_anchor,
            mask,
            track_id,
            instance_feature,
            anchor,
            time_interval,
            feature,
            spatial_shapes,
            level_start_index,
            image_wh,
            lidar2cam,
            cam_distortion,
            cam_intrinsic,
            aug_mat,
        ]
        outputs = [
            pred_instance_feature,
            pred_anchor,
            pred_class_score,
            pred_quality,
            pred_track_id,
        ]
        return inputs, outputs, first_frame

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        if (
            self.head.sampler.dn_metas is not None
            and self.head.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.head.sampler.dn_metas = None
        (
            instance_feature,  # (1, 900, 256) float32
            anchor,  # (1, 900, 11) float32
            temp_instance_feature,  # None
            temp_anchor,  # None
            time_interval,  # (1,)=0.5000 float32
        ) = self.head.instance_bank.get(
            batch_size, metas, dn_metas=self.head.sampler.dn_metas
        )

        """Inputs hook"""
        if self.first_frame:
            assert temp_instance_feature is None
            assert temp_anchor is None
            assert self.head.instance_bank.mask is None
            assert self.head.instance_bank.track_id is None

            self._temp_instance_feature = None
            self._temp_anchor = None
            self._mask = None
            self._track_id = None
            self.first_frame = False

        else:
            assert temp_instance_feature is not None
            assert temp_anchor is not None
            assert self.head.instance_bank.mask is not None
            assert self.head.instance_bank.track_id is not None

            self._temp_instance_feature = temp_instance_feature.clone()
            self._temp_anchor = temp_anchor.clone()
            self._mask = self.head.instance_bank.mask.clone()
            self.track_id = self.head.instance_bank.track_id.clone()

        self._instance_feature = instance_feature.clone()
        self._anchor = anchor.clone()
        self._time_interval = time_interval.clone()
        self._feature = feature_maps[0].clone()
        self._spatial_shapes = feature_maps[1].clone()
        self._level_start_index = feature_maps[2].clone()
        self._image_wh = metas["image_wh"].clone()
        # self._lidar2cam = metas["lidar2cam"].clone()
        # self._cam_distortion = metas["cam_distortion"].clone()
        # self._cam_intrinsic = metas["cam_intrinsic"].clone()
        # self._aug_mat = metas["aug_mat"].clone()

        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.head.training and hasattr(self.head.sampler, "get_dn_anchors"):
            if "track_id" in metas["img_metas"][0]:

                gt_track_id = [
                    torch.from_numpy(x["track_id"]).cuda() for x in metas["img_metas"]
                ]
            else:
                gt_track_id = None
            dn_metas = self.head.sampler.get_dn_anchors(
                metas[self.head.gt_cls_key],
                metas[self.head.gt_reg_key],
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
            attn_mask[num_free_instance:, num_free_instance:] = (
                dn_attn_mask  # (1120, 1120)
            )

        anchor_embed = self.head.anchor_encoder(anchor)  # (bs, 320+900, 256)
        if temp_anchor is not None:
            temp_anchor_embed = self.head.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        prediction = []
        classification = []
        quality = []
        for i, op in enumerate(self.head.operation_order):
            if self.head.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.head.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask if temp_instance_feature is None else None,
                )
            elif op == "gnn":
                instance_feature = self.head.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":  # [1, 900, 512] => [1, 900, 256]
                instance_feature = self.head.layers[i](instance_feature)
            elif op == "deformable":  # [1, 900, 256]
                # i = 0, 7
                instance_feature = self.head.layers[i](
                    instance_feature,  # [1, 900, 256]
                    anchor,  # [1, 900, 11]
                    anchor_embed,  # [1, 900, 256]
                    feature_maps,  # [[1, 89760, 256], [6, 4, 2], [6, 4, 4]]
                    metas,
                )
            elif op == "refine":
                anchor, cls, qt = self.head.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.head.training
                        or len(prediction) == self.head.num_single_frame_decoder - 1
                        or i == len(self.head.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == self.head.num_single_frame_decoder:
                    instance_feature, anchor = self.head.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None
                        and self.head.sampler.num_temp_dn_groups > 0  # default=3
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.head.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.head.instance_bank.num_anchor,
                            self.head.instance_bank.mask,  # None
                        )
                if i != len(self.head.operation_order) - 1:
                    # (1, 1220, 11) => (1, 1220, 256)
                    anchor_embed = self.head.anchor_encoder(anchor)
                if (
                    len(prediction) > self.head.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.head.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        """Onputs hook"""
        self._pred_instance_feature = instance_feature.clone()
        self._pred_anchor = anchor.clone()
        self._pred_class_score = cls.clone()
        self._pred_quality = qt.clone()
        if self.head.instance_bank.track_id is not None:
            self._pred_track_id = self.head.instance_bank.track_id.clone()

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

        self.head.instance_bank.cache(instance_feature, anchor, cls, metas)
        if not self.head.training:
            track_id = self.head.instance_bank.get_track_id(
                cls, self.head.decoder.score_threshold
            )
            output["track_id"] = track_id  # [1, 900], int64


def main():
    set_random_seed(seed=1, deterministic=True)

    args = parse_args()
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logger, console_handler, file_handler = set_logger(args.log, save_file=True)
    logger.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    cfg = read_cfg(args.config)  # dict

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg["data"]["test"]["test_mode"] = True

    # build the dataloader
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

    model.eval().cuda()

    backbone_hook = Sparse4D_backbone(model)
    head_hook = Sparse4D_head(model.head)

    for i, data in enumerate(data_loader):
        if i == 3:
            break
        with torch.no_grad():
            data = scatter(data, [0])[0]
            ori_imgs = data["ori_img"].detach().cpu().numpy()
            imgs = data["img"].detach().cpu().numpy()
            save_bins(
                inputs=[ori_imgs],
                outputs=[imgs],
                names=["ori_imgs", "imgs"],
                sample_index=i,
                logger=logger,
            )

            feature_maps = backbone_hook(img=data.pop("img"))
            logger.info(
                f"Start to save bin for Sparse4dBackbone, sampleindex={i} >>>>>>>>>>>>>>>>"
            )

            # save_bins_backbone(
            #     backbone_hook.io_hook()[0],
            #     backbone_hook.io_hook()[1],
            #     logger=logger,
            #     sample_index=i,
            # )

            _ = head_hook(feature_maps, data)
            # inputs, outputs, first_frame_flag = head_hook.io_hook()
            # if first_frame_flag:
            #     logger.info(
            #         f"Start to save bin for 1st frame Sparse4dHead, sampleindex={i} >>>>>>>>>>>>>>>>"
            #     )
            #     save_bins_1stframe_head(
            #         inputs,
            #         outputs,
            #         logger=logger,
            #         sample_index=i,
            #     )
            # else:
            #     logger.info(
            #         f"Start to save bin for frame > 1 Sparse4dHead, sampleindex={i} >>>>>>>>>>>>>>>>"
            #     )
            #     save_bins_head(
            #         inputs,
            #         outputs,
            #         logger=logger,
            #         sample_index=i,
            #     )


if __name__ == "__main__":
    main()

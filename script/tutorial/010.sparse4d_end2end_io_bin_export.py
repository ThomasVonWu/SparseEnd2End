# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import argparse
import torch
import torch.nn as nn
import logging
import numpy as np

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
    parser = argparse.ArgumentParser(description="Export each module bin file!")
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
        return [img], [feature]

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
        self._head = model
        self._first_frame = True

    def head_io_hook(
        self,
    ):
        """Head common input tensor names."""
        # feature = self._feature.detach().cpu().numpy()
        spatial_shapes = (
            self._spatial_shapes.int().detach().cpu().numpy()
        )  # int64->in32
        level_start_index = (
            self._level_start_index.int().detach().cpu().numpy()
        )  # int64->in32
        instance_feature = self._instance_feature.detach().cpu().numpy()
        anchor = self._anchor.detach().cpu().numpy()
        time_interval = self._time_interval.detach().cpu().numpy()
        image_wh = self._image_wh.detach().cpu().numpy()
        lidar2img = self._lidar2img.detach().cpu().numpy()

        """ Head common output tensor names. """
        pred_instance_feature = self._pred_instance_feature.detach().cpu().numpy()
        pred_anchor = self._pred_anchor.detach().cpu().numpy()
        pred_class_score = self._pred_class_score.detach().cpu().numpy()
        pred_quality_score = self._pred_quality_score.detach().cpu().numpy()

        if (
            self._temp_instance_feature is not None
            and self._temp_anchor is not None
            and self._mask is not None
            and self._track_id is not None
            and self._pred_track_id is not None
        ):
            """Head frame > 1 input tensor names."""
            temp_instance_feature = self._temp_instance_feature.detach().cpu().numpy()
            temp_anchor = self._temp_anchor.detach().cpu().numpy()
            mask = self._mask.int().detach().cpu().numpy()  # int64->in32
            track_id = self._track_id.int().detach().cpu().numpy()  # int64->in32

            """ Head frame > 1 output tensor names. """
            pred_track_id = (
                self._pred_track_id.int().detach().cpu().numpy()
            )  # int64->in32

        if self._first_frame:
            inputs = [
                # feature, # repeat with the backbone output
                spatial_shapes,
                level_start_index,
                instance_feature,
                anchor,
                time_interval,
                image_wh,
                lidar2img,
            ]
            outputs = [
                pred_instance_feature,
                pred_anchor,
                pred_class_score,
                pred_quality_score,
            ]
            return inputs, outputs

        inputs = [
            temp_instance_feature,
            temp_anchor,
            mask,
            track_id,
            instance_feature,
            anchor,
            time_interval,
            # feature, # repeat with the backbone output
            spatial_shapes,
            level_start_index,
            image_wh,
            lidar2img,
        ]
        outputs = [
            pred_instance_feature,
            pred_anchor,
            pred_class_score,
            pred_quality_score,
            pred_track_id,
        ]
        return inputs, outputs

    def instance_bank_io_hook(self):
        """InstanceBank::get() input tensor names."""
        ibank_timestamp = self._ibank_timestamp.detach().cpu().numpy()
        ibank_global2lidar = self._ibank_global2lidar.astype(
            np.float32
        )  # float64 -> float32

        """InstanceBank::get() output tensor names. """
        # self._instance_feature
        # self._anchor
        # self._time_interval
        # self._feature
        # self._spatial_shapes
        # sefl._level_start_index

        """InstanceBank::cache() input tensor names. """
        # self._pred_instance_feature
        # self._pred_instance_feature

        """InstanceBank::cache() output tensor names. """
        ibank_temp_confidence = self._ibank_temp_confidence.detach().cpu().numpy()
        ibank_confidence = self._ibank_confidence.detach().cpu().numpy()
        ibank_cached_feature = self._ibank_cached_feature.detach().cpu().numpy()
        ibank_cached_anchor = self._ibank_cached_anchor.detach().cpu().numpy()

        """InstanceBank GetTrackId() output tensor names. """
        ibank_prev_id = np.array(
            [self._ibank_prev_id.detach().cpu()], dtype=np.int32
        )  # int64 -> int32
        ibank_updated_cur_track_id = (
            self._ibank_updated_cur_track_id.int()
            .detach()
            .cpu()
            .numpy()  # int64 -> int32
        )
        ibank_updated_temp_track_id = (
            self._ibank_updated_temp_track_id.int()
            .detach()
            .cpu()
            .numpy()  # int64 -> int32
        )

        inputs = [
            ibank_timestamp,
            ibank_global2lidar,
        ]
        outputs = [
            ibank_temp_confidence,
            ibank_confidence,
            ibank_cached_feature,
            ibank_cached_anchor,
            ibank_prev_id,
            ibank_updated_cur_track_id,
            ibank_updated_temp_track_id,
        ]
        return inputs, outputs

    def post_process_io_hook(self):
        decoder_boxes_3d = self._decoder_boxes_3d.detach().cpu().numpy()
        decoder_scores_3d = self._decoder_scores_3d.detach().cpu().numpy()
        decoder_labels_3d = (
            self._decoder_labels_3d.int().detach().cpu().numpy()
        )  # int64 -> int32
        decoder_cls_scores = self._decoder_cls_scores.detach().cpu().numpy()
        decoder_track_ids = self._decoder_track_ids.int().detach().cpu().numpy()

        inputs = []
        outputs = [
            decoder_boxes_3d,
            decoder_scores_3d,
            decoder_labels_3d,
            decoder_cls_scores,
            decoder_track_ids,
        ]
        return inputs, outputs

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self._head.instance_bank.get(
            batch_size, metas, dn_metas=self._head.sampler.dn_metas
        )

        """InstanceBank::get() input hook. """
        self._ibank_timestamp = metas["timestamp"].clone()
        self._ibank_global2lidar = metas["img_metas"][0]["global2lidar"].copy()

        """Head input hook. """
        # self._feature = feature_maps[0].clone() # it repeats in backbone io_hook.
        self._spatial_shapes = feature_maps[1].clone()
        self._level_start_index = feature_maps[2].clone()

        self._instance_feature = instance_feature.clone()
        self._anchor = anchor.clone()
        self._time_interval = time_interval.clone()
        if self._first_frame:
            assert temp_instance_feature is None
            assert temp_anchor is None
            assert self._head.instance_bank.mask is None
            assert self._head.instance_bank.track_id is None

            self._temp_instance_feature = None
            self._temp_anchor = None
            self._mask = None
            self._track_id = None
        else:
            assert temp_instance_feature is not None
            assert temp_anchor is not None
            assert self._head.instance_bank.mask is not None
            assert self._head.instance_bank.track_id is not None

            self._temp_instance_feature = temp_instance_feature.clone()
            self._temp_anchor = temp_anchor.clone()
            self._mask = self._head.instance_bank.mask.clone()
            self._track_id = self._head.instance_bank.track_id.clone()

        self._image_wh = metas["image_wh"].clone()
        self._lidar2img = metas["lidar2img"].clone()

        attn_mask = None
        temp_dn_reg_target = None

        anchor_embed = self._head.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self._head.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        prediction = []
        classification = []
        quality = []
        for i, op in enumerate(self._head.operation_order):
            if self._head.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self._head.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask if temp_instance_feature is None else None,
                )
            elif op == "gnn":
                instance_feature = self._head.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self._head.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self._head.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif op == "refine":
                anchor, cls, qt = self._head.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self._head.training
                        or len(prediction) == self._head.num_single_frame_decoder - 1
                        or i == len(self._head.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == self._head.num_single_frame_decoder:
                    instance_feature, anchor = self._head.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                if i != len(self._head.operation_order) - 1:
                    anchor_embed = self._head.anchor_encoder(anchor)
                if (
                    len(prediction) > self._head.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self._head.instance_bank.num_temp_instances
                    ]

        """Head output hook. """
        self._pred_instance_feature = instance_feature.clone()
        self._pred_anchor = anchor.clone()
        self._pred_class_score = cls.clone()
        self._pred_quality_score = qt.clone()
        if self._head.instance_bank.track_id is not None:
            self._pred_track_id = self._head.instance_bank.track_id.clone()

        output = {}
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )

        self._head.instance_bank.cache(instance_feature, anchor, cls, metas)

        """InstanceBank::cache() output hook. """
        self._ibank_temp_confidence = self._head.instance_bank.temp_confidence.clone()
        self._ibank_confidence = self._head.instance_bank.confidence.clone()
        self._ibank_cached_feature = self._head.instance_bank.cached_feature.clone()
        self._ibank_cached_anchor = self._head.instance_bank.cached_anchor.clone()

        track_id = self._head.instance_bank.get_track_id(
            cls, self._head.decoder.score_threshold
        )
        output["track_id"] = track_id  # [1, 900], int64

        """InstanceBank::get_track_id() output hook. """
        self._ibank_prev_id = self._head.instance_bank.prev_id.clone()
        self._ibank_updated_cur_track_id = track_id.clone()
        self._ibank_updated_temp_track_id = self._head.instance_bank.track_id.clone()

        """Postprocessor output  hook. """
        output = self._head.decoder.decode(
            output["classification"],
            output["prediction"],
            output.get("track_id"),
            output.get("quality"),
            output_idx=-1,
        )[batch_size - 1]
        self._decoder_boxes_3d = output["boxes_3d"]
        self._decoder_scores_3d = output["scores_3d"]
        self._decoder_labels_3d = output["labels_3d"]
        self._decoder_cls_scores = output["cls_scores"]
        self._decoder_track_ids = output["track_ids"]


def main():
    set_random_seed(seed=1, deterministic=True)

    args = parse_args()
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logger, console_handler, file_handler = set_logger(args.log, save_file=True)
    logger.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

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
            save_bins(
                inputs=backbone_hook.io_hook()[0],
                outputs=backbone_hook.io_hook()[1],
                names=["imgs", "feature"],
                sample_index=i,
                logger=logger,
            )

            head_hook(feature_maps, data)
            inputs, outputs = head_hook.head_io_hook()
            if head_hook._first_frame:
                head_hook._first_frame = False
                logger.info(
                    f"Start to save bin for first frame Sparse4dHead, sampleindex={i} >>>>>>>>>>>>>>>>"
                )
                save_bins(
                    inputs=inputs,
                    outputs=outputs,
                    names=[
                        "spatial_shapes",
                        "level_start_index",
                        "instance_feature",
                        "anchor",
                        "time_interval",
                        "image_wh",
                        "lidar2img",
                        "pred_instance_feature",
                        "pred_anchor",
                        "pred_class_score",
                        "pred_quality_score",
                    ],
                    logger=logger,
                    sample_index=i,
                )
            else:
                logger.info(
                    f"Start to save bin for frame > 1 Sparse4dHead, sampleindex={i} >>>>>>>>>>>>>>>>"
                )
                save_bins(
                    inputs=inputs,
                    outputs=outputs,
                    names=[
                        "temp_instance_feature",
                        "temp_anchor",
                        "mask",
                        "track_id",
                        "instance_feature",
                        "anchor",
                        "time_interval",
                        "spatial_shapes",
                        "level_start_index",
                        "image_wh",
                        "lidar2img",
                        "pred_instance_feature",
                        "pred_anchor",
                        "pred_class_score",
                        "pred_quality_score",
                        "pred_track_id",
                    ],
                    logger=logger,
                    sample_index=i,
                )

            logger.info(
                f"Start to save bin for InstanceBank, sampleindex={i} >>>>>>>>>>>>>>>>"
            )
            save_bins(
                inputs=head_hook.instance_bank_io_hook()[0],
                outputs=head_hook.instance_bank_io_hook()[1],
                names=[
                    "ibank_timestamp",
                    "ibank_global2lidar",
                    "ibank_temp_confidence",
                    "ibank_confidence",
                    "ibank_cached_feature",
                    "ibank_cached_anchor",
                    "ibank_prev_id",
                    "ibank_updated_cur_track_id",
                    "ibank_updated_temp_track_id",
                ],
                logger=logger,
                sample_index=i,
            )

            logger.info(
                f"Start to save bin for Postprocessor, sampleindex={i} >>>>>>>>>>>>>>>>"
            )
            save_bins(
                inputs=head_hook.post_process_io_hook()[0],
                outputs=head_hook.post_process_io_hook()[1],
                names=[
                    "decoder_boxes_3d",
                    "decoder_scores_3d",
                    "decoder_labels_3d",
                    "decoder_cls_scores",
                    "decoder_track_ids",
                ],
                logger=logger,
                sample_index=i,
            )


if __name__ == "__main__":
    main()

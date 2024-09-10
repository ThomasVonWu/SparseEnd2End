# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import logging
import argparse

import onnx
from onnxsim import simplify

import torch
from torch import nn
from typing import Optional, Dict, Any

from modules.sparse4d_detector import *
from tool.utils.logger import set_logger
from tool.utils.config import read_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy SparseEND2END Backbone!")
    parser.add_argument(
        "--cfg",
        type=str,
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
        help="deploy config file path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt/sparse4dv3_r50.pth",
        help="deploy ckpt path",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="deploy/onnx/export_backbone_onnx.log",
    )
    parser.add_argument(
        "--save_onnx",
        type=str,
        default="deploy/onnx/sparse4dbackbone.onnx",
    )
    args = parser.parse_args()
    return args


class Sparse4DBackbone(nn.Module):
    def __init__(self, model):
        super(Sparse4DBackbone, self).__init__()
        self.model = model

    def forward(self, img):

        feature, spatial_shapes, level_start_index = self.model.extract_feat(img)

        return feature


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_onnx), exist_ok=True)
    logger, console_handler, file_handler = set_logger(args.log, True)
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    logger.info("Export Sparse4d Backbone Onnx...")

    cfg = read_cfg(args.cfg)
    model = build_module(cfg["model"])
    checkpoint = args.ckpt
    _ = model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)
    model.eval()

    BS = 1
    NUMS_CAM = 6
    C = 3
    INPUT_H = 256
    INPUT_W = 704
    dummy_img = torch.randn(BS, NUMS_CAM, C, INPUT_H, INPUT_W).cuda()

    backbone = Sparse4DBackbone(model).cuda()
    with torch.no_grad():
        torch.onnx.export(
            backbone,
            (dummy_img,),
            args.save_onnx,
            input_names=["img"],
            output_names=[
                "feature",
            ],
            opset_version=15,
            do_constant_folding=True,
            verbose=False,
        )

        onnx_orig = onnx.load(args.save_onnx)
        onnx_simp, check = simplify(onnx_orig)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_simp, args.save_onnx)
        logger.info(f'🚀 Export onnx completed. ONNX saved in "{args.save_onnx}" 🤗.')

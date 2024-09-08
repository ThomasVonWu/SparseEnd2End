# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import torch
import argparse
import numpy as np
import tensorrt as trt
from cuda import cudart

import logging
from tool.utils.logger import set_logger

from modules.sparse4d_detector import *
from typing import Optional, Dict, Any
from tool.utils.config import read_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Onnx Model Inference Consistency Check!"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt/sparse4dv3_r50.pth",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="deploy/onnx/trt_consistencycheck.log",
    )
    parser.add_argument(
        "--trtengine",
        type=str,
        default="deploy/engine/sparse4dbackbone.engine",
    )
    args = parser.parse_args()
    return args


def build_network(trtFile):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            return
    return engine


def inference(
    engine,
    input,
    trt_old,
    logger,
):
    if trt_old:
        nIO = engine.num_bindings
        lTensorName = [engine.get_binding_name(i) for i in range(nIO)]
        nInput = sum([engine.binding_is_input(lTensorName[i]) for i in range(nIO)])

        bufferH = []
        bufferH.append(input)
        for i in range(nInput, nIO):
            bufferH.append(
                np.zeros(
                    engine.get_binding_shape(lTensorName[i]),
                    dtype=trt.nptype(engine.get_binding_dtype(lTensorName[i])),
                )
            )

        for j in range(nIO):
            logger.debug(
                f"Engine Binding name:{lTensorName[j]}, shape:{engine.get_binding_shape}, type:{engine.get_binding_dtype} ."
            )
            logger.debug(
                f"Compared Input Data{lTensorName[j]} shape:{bufferH[j].shape}, shape:{bufferH[j].type}, shape:{bufferH[j].nbytes} ."
            )

    else:
        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(
            trt.TensorIOMode.INPUT
        )

        bufferH = []
        bufferH.append(input)
        context = engine.create_execution_context()
        for i in range(nInput, nIO):
            bufferH.append(
                np.zeros(
                    context.get_tensor_shape(lTensorName[i]),
                    dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i])),
                )
            )

        for j in range(nIO):
            logger.debug(
                f"Engine Binding name:{lTensorName[j]}, shape:{context.get_tensor_shape(lTensorName[j])}, type:{trt.nptype(engine.get_tensor_dtype(lTensorName[j]))} ."
            )
            logger.debug(
                f"Compared Input Data:{lTensorName[j]} shape:{bufferH[j].shape}, type:{bufferH[j].dtype} ."
            )

    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(
            bufferD[i],
            bufferH[i].ctypes.data,
            bufferH[i].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

    if trt_old:
        binding_addrs = [int(bufferD[i]) for i in range(nIO)]
        context.execute_v2(binding_addrs)
    else:
        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(
            bufferH[i].ctypes.data,
            bufferD[i],
            bufferH[i].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )

    for b in bufferD:
        cudart.cudaFree(b)

    return bufferH[-1]


def printArrayInformation(x, logger, info: str):
    logger.debug(f"Name={info}")
    logger.debug(
        "\tSumAbs=%.3f, Max=%.3f, Min=%.3f"
        % (
            np.sum(abs(x)),
            np.max(x),
            np.min(x),
        )
    )


def trt_inference(
    engine,
    input,
    trt_old,
    logger,
):
    output = inference(
        engine,
        input,
        trt_old,
        logger,
    )

    return output


def model_infer(model, dummy_img):
    with torch.no_grad():
        feature, spatial_shapes, level_start_index = model.extract_feat(dummy_img)
    return feature.cpu().numpy()


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def main():
    args = parse_args()
    logger, _, _ = set_logger(args.log, save_file=False)
    logger.setLevel(logging.DEBUG)

    logger.info("Sparse4d Backbone Onnx Inference Consistency Check!...")

    cfg = read_cfg(args.cfg)
    model = build_module(cfg["model"])
    checkpoint = args.ckpt
    _ = model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)

    trt_engine = build_network(args.trtengine)

    for i in range(3):
        np.random.seed(i)
        logger.debug(f"Test Sample {i} Results:")

        BS = 1
        NUM_CAMS = 6
        C = 3
        H = 256
        W = 704
        dummy_img = np.random.rand(BS, NUM_CAMS, C, H, W).astype(np.float32)

        output1 = model_infer(model.cuda().eval(), torch.from_numpy(dummy_img).cuda())

        output2 = trt_inference(trt_engine, dummy_img, False, logger)

        cosine_distance = 1 - np.dot(output1.flatten(), output2.flatten()) / (
            np.linalg.norm(output1.flatten()) * np.linalg.norm(output2.flatten())
        )
        assert (
            cosine_distance < 1e-3
        ), f"Error in cosine_distance = {float(cosine_distance)} !"
        logger.info(f"cosine_distance = {float(cosine_distance)}")

        max_abs_distance = float((np.abs(output1 - output2)).max())
        assert (
            max_abs_distance < 0.1
        ), f"Error in max_abs_distance = {max_abs_distance} !"
        logger.info(f"max(abs(a-b))   = {max_abs_distance}")


if __name__ == "__main__":
    main()

# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import ctypes
import numpy as np
import tensorrt as trt

from typing import List
from cuda import cudart
from tool.utils.logger import logger_wrapper


def read_bin(samples=1):
    data = list()
    for i in range(samples):
        shape1 = [1, 89760, 256]
        feature = np.fromfile(
            f"deploy/dfa_plugin/asset/sample_{i}_rand_fetaure_1x89760x256_float32.bin",
            dtype=np.float32,
        ).reshape(shape1)

        shape2 = [6, 4, 2]
        spatial_shapes = np.fromfile(
            f"deploy/dfa_plugin/asset/sample_{i}_rand_spatial_shapes_6x4x2_int32.bin",
            dtype=np.int32,
        ).reshape(shape2)

        shape3 = [6, 4]
        level_start_index = np.fromfile(
            f"deploy/dfa_plugin/asset/sample_{i}_rand_level_start_index_6x4_int32.bin",
            dtype=np.int32,
        ).reshape(shape3)

        shape4 = [1, 900, 13, 6, 2]
        sample_loc = np.fromfile(
            f"deploy/dfa_plugin/asset/sample_{i}_rand_sampling_loc_1x900x13x6x2_float32.bin",
            dtype=np.float32,
        ).reshape(shape4)

        shape5 = [1, 900, 13, 6, 4, 8]
        weights = np.fromfile(
            f"deploy/dfa_plugin/asset/sample_{i}_rand_weights_1x900x13x6x4x8_float32.bin",
            dtype=np.float32,
        ).reshape(shape5)

        shape6 = [1, 900, 256]
        output = np.fromfile(
            f"deploy/dfa_plugin/asset/sample_{i}_output_1x900x256_float32.bin",
            dtype=np.float32,
        ).reshape(shape6)

        data.append(
            [feature, spatial_shapes, level_start_index, sample_loc, weights, output]
        )
    return data


def getPlugin(plugin_name) -> trt.tensorrt.IPluginV2:
    for i, c in enumerate(trt.get_plugin_registry().plugin_creator_list):
        logger.debug(f"We have plugin{i} : {c.name}")
        if c.name == plugin_name:
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))


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
    feature,
    spatial_shapes,
    scale_start_index,
    sampling_loc,
    weights,
    output,
    trt_old,
    logger,
):
    if trt_old:
        nIO = engine.num_bindings
        lTensorName = [engine.get_binding_name(i) for i in range(nIO)]
        nInput = sum([engine.binding_is_input(lTensorName[i]) for i in range(nIO)])

        bufferH = []
        bufferH.append(feature)
        bufferH.append(spatial_shapes)
        bufferH.append(scale_start_index)
        bufferH.append(sampling_loc)
        bufferH.append(weights)
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
        nIO = engine.num_io_tensors  # 6
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(
            trt.TensorIOMode.INPUT
        )

        bufferH = []
        bufferH.append(feature)
        bufferH.append(spatial_shapes)
        bufferH.append(scale_start_index)
        bufferH.append(sampling_loc)
        bufferH.append(weights)
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

    for i in range(nInput):
        printArrayInformation(bufferH[i], logger, info=f"{lTensorName[i]}")
    printArrayInformation(bufferH[nInput], logger, info=f"{lTensorName[i]}")
    printArrayInformation(output, logger, info="expect-output")

    return nInput, nIO, bufferH


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


def dfa_test(
    feature,
    spatial_shapes,
    level_start_index,
    sampling_loc,
    weights,
    output,
    trt_old,
    logger,
    plugin_name,
    soFile,
    trtFile,
):

    ctypes.cdll.LoadLibrary(soFile)
    plugin = getPlugin(plugin_name)
    logger.info("Test '%s'" % (plugin.plugin_type))

    engine = build_network(trtFile)
    if engine == None:
        logger.error(f"{plugin_name} Engine Building Failed: {trtFile} !")
        return

    nInput, nIO, bufferH = inference(
        engine,
        feature,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        weights,
        output,
        trt_old,
        logger,
    )
    output2 = bufferH[-1]
    cosine_distance = 1 - np.dot(output.flatten(), output2.flatten()) / (
        np.linalg.norm(output.flatten()) * np.linalg.norm(output2.flatten())
    )
    assert (
        cosine_distance < 1e-3
    ), f"Error in cosine_distance = {float(cosine_distance)} !"
    logger.info(f"[DeformableAttentionAggrPlugin] cosine_distance = {float(cosine_distance)}")

    max_abs_distance = float((np.abs(output - output2)).max())
    assert max_abs_distance < 0.1, f"Error in max_abs_distance = {max_abs_distance} !"
    logger.info(f"[DeformableAttentionAggrPlugin] max(abs(a-b))   = {max_abs_distance}")

def main(
    data: List,
    trt_old,
    logger,
    plugin_name="DeformableAttentionAggrPlugin",
    soFile="deploy/dfa_plugin/lib/deformableAttentionAggr.so",
    trtFile="deploy/dfa_plugin/engine/deformableAttentionAggr.engine",
):
    for x in data:
        feature, spatial_shapes, level_start_index, sampling_loc, weights, output = x
        dfa_test(
            feature,
            spatial_shapes,
            level_start_index,
            sampling_loc,
            weights,
            output,
            trt_old,
            logger,
            plugin_name,
            soFile,
            trtFile,
        )


if __name__ == "__main__":
    from tool.utils.logger import logger_wrapper

    logger, _, _ = logger_wrapper("", False)

    a, b, c, d = (trt.__version__).split(".")
    verson = eval(a + "." + b)
    logger.info("Python Tensor Version is: {trt.__version__} !")

    if verson < 8.5:
        trt_old = True
    else:
        trt_old = False

    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    np.random.seed(1)

    logger.info("Starting unit test...")
    data = read_bin(1)
    main(data, trt_old, logger)
    logger.info("All tests are passed!!!")

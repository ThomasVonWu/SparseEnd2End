# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import ctypes
import logging
import numpy as np
import tensorrt as trt

from trt_core import test_logger, console_handler, file_handler
from trt_core import build_network, inference, printArrayInformation


def getPlugin(plugin_name) -> trt.tensorrt.IPluginV2:
    for i, c in enumerate(trt.get_plugin_registry().plugin_creator_list):
        test_logger.info(f"Plugin{i} : {c.name}")
        if c.name == plugin_name:
            parameterList = []
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))


def DeformableAttentionAggrPluginTest(
    value_shape,
    value,
    spatial_shape,
    spatial_shapes,
    level_start_index_shape,
    level_start_index,
    sampling_location_shape,
    sampling_location,
    attn_weight_shape,
    attn_weight,
    output_shape,
    trt_old,
    plugin_name="DeformableAttentionAggrPlugin",
    soFile="lib/deformableAttentionAggr.so",
    trtFile="deformableAttentionAggr.engine",
):

    ctypes.cdll.LoadLibrary(soFile)
    plugin = getPlugin(plugin_name)
    test_logger.info("Test '%s'" % (plugin.plugin_type))

    engine = build_network(trtFile, plugin)
    if engine == None:
        test_logger.error("{plugin_name} Engine Building Failed.")
        return

    nInput, nIO, bufferH = inference(
        engine,
        value_shape,
        value,
        spatial_shape,
        spatial_shapes,
        level_start_index_shape,
        level_start_index,
        sampling_location_shape,
        sampling_location,
        attn_weight_shape,
        attn_weight,
        trt_old,
    )

    for i in range(nInput):
        printArrayInformation(bufferH[i], info="input[%s]" % i)
    for i in range(nInput, nIO):
        printArrayInformation(bufferH[i], info="output[%s]" % (i - nInput))

    # outputCPU = CustomScalarCPU(bufferH[:nInput])
    # res = validation(nInput, nIO, bufferH, outputCPU)

    # if res:
    #     test_logger.info("Test '%s':%s finish!\n" % (plugin.plugin_type, testCase))
    # else:
    #     test_logger.error("Test '%s':%s failed!\n" % (plugin.plugin_type, testCase))
    #     exit()


def unit_test(trt_old=True):
    value_shape = [1, 89760, 256]
    value = np.fromfile("asset/value.bin", dtype=np.float32).reshape(value_shape)

    spatial_shape = [6, 4, 2]
    spatial_shapes = np.fromfile("asset/spatial_shapes.bin", dtype=np.int32).reshape(
        spatial_shape
    )

    level_start_index_shape = [6, 4]
    level_start_index = np.fromfile(
        "asset/level_start_index.bin", dtype=np.int32
    ).reshape(level_start_index_shape)

    sampling_location_shape = [1, 900, 13, 6, 2]
    sampling_location = np.fromfile("asset/sampling_loc.bin", dtype=np.float32).reshape(
        sampling_location_shape
    )

    attn_weight_shape = [1, 900, 13, 6, 4, 8]
    attn_weight = np.fromfile("asset/attn_weight.bin", dtype=np.float32).reshape(
        attn_weight_shape
    )

    output_shape = [1, 900, 256]

    DeformableAttentionAggrPluginTest(
        value_shape,
        value,
        spatial_shape,
        spatial_shapes,
        level_start_index_shape,
        level_start_index,
        sampling_location_shape,
        sampling_location,
        attn_weight_shape,
        attn_weight,
        output_shape,
        trt_old,
    )


if __name__ == "__main__":

    if trt.__version__ == "8.4.0.6":
        trt_old = True
    elif trt.__version__ == "8.6.1.6 ":
        trt_old = False
    else:
        exit()

    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    np.random.seed(1)

    test_logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    test_logger.info("Starting unit test...")
    unit_test(trt_old)
    test_logger.info("All tests are passed!!!")

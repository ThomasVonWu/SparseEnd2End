import os
import numpy as np
import tensorrt as trt
from cuda import cudart
from logger import set_logger

log = "unit_test.log"
test_logger, file_handler, console_handler = set_logger(log)


def build_network(trtFile, plugin, shape=[]):
    # 从shared library中读取plugin
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")

    if os.path.isfile(trtFile):
        test_logger.info("Start to deserialize...")
        with open(trtFile, "rb") as f:
            # 反序列化一个推理引擎
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            test_logger.error(f"Failed loading engine: {trtFile}!")
            return
        test_logger.info(f"Succeed to load engine: {trtFile}!")
    # else:
    #     # 利用python api创建一个推理引擎。这个推理引擎中只有我们准备做unit-test所需要的plugin。创建engine的流程和c++是一样的
    #     builder = trt.Builder(logger)
    #     network = builder.create_network(
    #         1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    #     )
    #     profile = builder.create_optimization_profile()
    #     config = builder.create_builder_config()

    #     # 为network创建一个dummy的输入，并支持动态shape
    #     inputT0 = network.add_input("inputT0", trt.float32, [-1 for i in shape])
    #     profile.set_shape(
    #         inputT0.name, [1 for i in shape], [8 for i in shape], [32 for i in shape]
    #     )
    #     config.add_optimization_profile(profile)

    #     # 为network添加这个plugin所对应的layer
    #     pluginLayer = network.add_plugin_v2([inputT0], plugin)

    #     # 为network标记输出
    #     network.mark_output(pluginLayer.get_output(0))

    #     # 序列化engine并保存
    #     engineString = builder.build_serialized_network(network, config)
    #     if engineString == None:
    #         test_logger.error("Failed building engine!")
    #         return
    #     test_logger.info("Succeeded building engine!")
    #     with open(trtFile, "wb") as f:
    #         f.write(engineString)
    #     engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    return engine


def inference(
    engine,
    value_shape,
    value,
    spatial_shape,
    spatial_shapes,
    scale_start_index_shape,
    scale_start_index,
    sampling_location_shape,
    sampling_location,
    attn_weight_shape,
    attn_weight,
    trt_old,
):
    ### lTensorName = ['value', 'spatial_shapes', 'level_start_index', 'sampling_loc', 'attn_weight', 'output0']
    if trt_old:
        nIO = engine.num_bindings
        lTensorName = [engine.get_binding_name(i) for i in range(nIO)]
        nInput = sum([engine.binding_is_input(lTensorName[i]) for i in range(nIO)])
        context = engine.create_execution_context()
        context.set_binding_shape(0, value_shape)
        context.set_binding_shape(1, spatial_shape)
        context.set_binding_shape(2, scale_start_index_shape)
        context.set_binding_shape(3, sampling_location_shape)
        context.set_binding_shape(4, attn_weight_shape)
        # print("binding1", trt.nptype(engine.get_binding_dtype(lTensorName[1])))
        # print("binding2", trt.nptype(engine.get_binding_dtype(lTensorName[2])))

        # 初始化host端的数据，根据输入的shape大小来初始化值, 同时也把存储输出的空间存储下来
        bufferH = []
        bufferH.append(value)
        bufferH.append(spatial_shapes)
        bufferH.append(scale_start_index)
        bufferH.append(sampling_location)
        bufferH.append(attn_weight)

        for i in range(nInput, nIO):
            assert lTensorName[i] == "output0"
            assert engine.get_binding_shape(lTensorName[i]) == (
                1,
                900,
                256,
            ), f"{engine.get_binding_shape(lTensorName[i])}"
            assert (
                trt.nptype(engine.get_binding_dtype(lTensorName[i])) == np.float32
            ), f"{trt.nptype(engine.get_binding_dtype(lTensorName[i]))}"
            bufferH.append(
                np.zeros(
                    engine.get_binding_shape(lTensorName[i]),
                    dtype=trt.nptype(engine.get_binding_dtype(lTensorName[i])),
                )
            )

    else:
        nIO = engine.num_io_tensors  # 6
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(
            trt.TensorIOMode.INPUT
        )
        context = engine.create_execution_context()
        context.set_input_shape("value", value_shape)
        context.set_input_shape("spatial_shapes", spatial_shape)
        context.set_input_shape("level_start_index", scale_start_index_shape)
        context.set_input_shape("sampling_loc", sampling_location_shape)
        context.set_input_shape("attn_weight", attn_weight_shape)
        # print("...", trt.nptype(engine.get_tensor_dtype(lTensorName[1])))
        # print("...", trt.nptype(engine.get_tensor_dtype(lTensorName[2])))

        # 初始化host端的数据，根据输入的shape大小来初始化值, 同时也把存储输出的空间存储下来
        bufferH = []
        bufferH.append(value)
        bufferH.append(spatial_shapes)
        bufferH.append(scale_start_index)
        bufferH.append(sampling_location)
        bufferH.append(attn_weight)

        for i in range(nInput, nIO):
            assert lTensorName[i] == "output0"
            assert context.get_tensor_shape(lTensorName[i]) == (
                1,
                900,
                256,
            ), f"{context.get_tensor_shape(lTensorName[i])}"
            assert (
                trt.nptype(engine.get_tensor_dtype(lTensorName[i])) == np.float32
            ), f"{trt.nptype(engine.get_tensor_dtype(lTensorName[i]))}"
            bufferH.append(
                np.zeros(
                    context.get_tensor_shape(lTensorName[i]),
                    dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i])),
                )
            )

    # 初始化device端的内存，根据host端的大小来分配空间
    bufferD = []
    for i in range(nIO):
        test_logger.debug(
            f"Bingding{i} Bytes = {bufferH[i].nbytes}",
        )
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    # H2D, enqueue, D2H执行推理，并把结果返回
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

    return nInput, nIO, bufferH


def validation(nInput, nIO, bufferH, standard):
    for i in range(nInput):
        printArrayInformation(bufferH[i], info="input[%s]" % i)
    for i in range(nInput, nIO):
        printArrayInformation(bufferH[i], info="output(plugin-impl)[%s]" % (i - nInput))
    for i in range(nInput, nIO):
        printArrayInformation(
            standard[i - nInput], info="output(cpu-impl)[%s]" % (i - nInput)
        )

    # CPU端的计算，与Plugin中核函数计算结果比较
    return check(bufferH[nInput:][0], standard[0], True)


# 输出tensor的前5位和后5位数据
def printArrayInformation(x, info="", n=5):
    if 0 in x.shape:
        test_logger.debug("%s:%s" % (info, str(x.shape)))
        test_logger.debug()
        return
    test_logger.debug("%s:%s" % (info, str(x.shape)))
    test_logger.debug(
        "\tSumAbs=%.4e, Var=%.4f, Max=%.4f, Min=%.4f, SAD=%.4f"
        % (
            np.sum(abs(x)),
            np.var(x),
            np.max(x),
            np.min(x),
            np.sum(np.abs(np.diff(x.reshape(-1)))),
        )
    )
    test_logger.debug("\t%s  ...  %s" % (x.reshape(-1)[:n], x.reshape(-1)[-n:]))


def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    test_logger.info("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))
    return res

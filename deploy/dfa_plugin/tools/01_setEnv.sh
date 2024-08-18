#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.


################***EnvVersion-1***#################
###     LinuxPlatform   :     x86_64            ###
###     TensorRT        :     8.4.0.6           ###
###     CUDA            :     11.3              ###
###     cuDNN           :     8.4.1.5           ###
###     CUDA capability :     sm_86             ###
################***EnvVersion-2***#################


################***EnvVersion-2***#################
###     LinuxPlatform   :     x86_64            ###
###     TensorRT        :     8.6.1.6           ###
###     CUDA            :     11.6              ###
###     cuDNN           :     8.4.1.5           ###
###     CUDA capability :     sm_86             ###
################***EnvVersion-1***#################



EnvVersion=1
if [ $EnvVersion = 1 ]; then
    export ENV_TensorRT_LIB=~/.cache/bazel/_bazel_wxc/fe1d1d62952da7ed96f34d290e1a4680/external/tensorrt_amd64/usr/lib/x86_64-linux-gnu
    export ENV_TensorRT_INC=~/.cache/bazel/_bazel_wxc/fe1d1d62952da7ed96f34d290e1a4680/external/tensorrt_amd64/usr/include/x86_64-linux-gnu
    export ENV_TensorRT_BIN=~/.cache/bazel/_bazel_wxc/fe1d1d62952da7ed96f34d290e1a4680/external/tensorrt_amd64/usr/src/tensorrt/bin
    export ENV_CUDA_LIB=/usr/local/cuda-11.3/lib64
    export ENV_CUDA_INC=/usr/local/cuda-11.3/include
    export ENV_CUDA_BIN=/usr/local/cuda-11.3/bin
else
    export ENV_TensorRT_LIB=/mnt/data/env_cfg/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8/TensorRT-8.6.1.6/lib
    export ENV_TensorRT_INC=/mnt/data/env_cfg/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8/TensorRT-8.6.1.6/include
    export ENV_TensorRT_BIN=/mnt/data/env_cfg/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8/TensorRT-8.6.1.6/bin
    export ENV_CUDA_LIB=/usr/local/cuda-11.6/lib64
    export ENV_CUDA_INC=/usr/local/cuda-11.6/include
    export ENV_CUDA_BIN=/usr/local/cuda-11.6/bin
fi

export ENV_cuDNN_LIB=/mnt/data/env_cfg/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib

if [ ! -f "${ENV_TensorRT_BIN}/trtexec" ]; then
    echo "[ERROR] Failed to Find ${ENV_TensorRT_BIN}/trtexec!"
    return
fi

if [ ! -f "${ENV_CUDA_BIN}/nvcc" ]; then
    echo "[ERROR] Failed to Find ${ENV_CUDA_BIN}/nvcc!"
    return
fi

if [ -f "tools/cudasm.sh" ]; then
    . "tools/cudasm.sh"
else
    echo "[ERROR] Failed to Find \"dfa_plugin/tools/cudasm.sh\" File!"
    return
fi

echo "===================================================================================================================="
echo "||  Config Environment Below:"
echo "||  TensorRT LIB : $ENV_TensorRT_LIB"
echo "||  TensorRT INC : $ENV_TensorRT_INC"
echo "||  TensorRT BIN : $ENV_TensorRT_BIN"
echo "||  CUDA     LIB : $ENV_CUDA_LIB"
echo "||  CUDA     INC : $ENV_CUDA_INC"
echo "||  CUDA     BIN : $ENV_CUDA_BIN"
echo "||  CUDNN    LIB : $ENV_cuDNN_LIB"
echo "||  CUDASM       : sm_$cudasm"
echo "===================================================================================================================="

export CUDASM=$cudasm
export PATH=$ENV_TensorRT_BIN:$CUDA_BIN:$PATH
export LD_LIBRARY_PATH=$ENV_TensorRT_LIB:$ENV_CUDA_LIB:$ENVcuDNN_LIB:$LD_LIBRARY_PATH

export ENVBUILDDIR=build
export ENVTARGETPLUGIN=lib/deformableAttentionAggr.so
export ENVONNX=onnx/deformableAttentionAggr.onnx
export ENVTRTLOGSDIR=trtlogs
export ENVEINGINENAME=deformableAttentionAggr.engine


echo "[INFO] Config Env Done, Please Check EnvPrintOut Above!"
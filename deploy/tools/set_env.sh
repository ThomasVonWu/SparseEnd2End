#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

################***EnvVersion-1***#################
###     LinuxPlatform:            x86_64                                                        ###
###     TensorRT :                    8.5.1.7                                                          ###
###     CUDA :                             11.6                                                            ###
###     cuDNN:                          8.6.0.163                                                    ###
###     CUDA capability :        sm_86                                                         ###
################***EnvVersion-1***#################

################***EnvVersion-2***#################
###     LinuxPlatform:            x86_64                                                        ###
###     TensorRT :                    8.6.1.6                                                          ###
###     CUDA :                             11.6                                                            ###
###     cuDNN:                          8.6.0.163                                                    ###
###     CUDA capability :        sm_86                                                         ###
################***EnvVersion-2***#################

EnvVersion=2
if [ $EnvVersion = 1 ]; then
    export ENV_TensorRT_LIB=/mnt/env/tensorrt/TensorRT-8.5.1.7/lib
    export ENV_TensorRT_INC=/mnt/env/tensorrt/TensorRT-8.5.1.7/include
    export ENV_TensorRT_BIN=/mnt/env/tensorrt/TensorRT-8.5.1.7/bin
    export ENV_CUDA_LIB=/usr/local/cuda-11.6/lib64
    export ENV_CUDA_INC=/usr/local/cuda-11.6/include
    export ENV_CUDA_BIN=/usr/local/cuda-11.6/bin
else
    export ENV_TensorRT_LIB=/mnt/env/tensorrt/TensorRT-8.6.1.6/lib
    export ENV_TensorRT_INC=/mnt/env/tensorrt/TensorRT-8.6.1.6/include
    export ENV_TensorRT_BIN=/mnt/env/tensorrt/TensorRT-8.6.1.6/bin
    export ENV_CUDA_LIB=/usr/local/cuda-11.6/lib64
    export ENV_CUDA_INC=/usr/local/cuda-11.6/include
    export ENV_CUDA_BIN=/usr/local/cuda-11.6/bin
fi
export ENV_cuDNN_LIB=/mnt/env/tensorrt/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib

if [ ! -f "${ENV_TensorRT_BIN}/trtexec" ]; then
    echo "[ERROR] Failed to Find ${ENV_TensorRT_BIN}/trtexec!"
    return
fi

if [ ! -f "${ENV_CUDA_BIN}/nvcc" ]; then
    echo "[ERROR] Failed to Find ${ENV_CUDA_BIN}/nvcc!"
    return
fi

if [ -f "deploy/dfa_plugin/tools/cudasm.sh" ]; then
    . "deploy/dfa_plugin/tools/cudasm.sh"
else
    echo "[ERROR] Failed to Find \"deploy/dfa_plugin/tools/cudasm.sh\" File!"
    return
fi

# Part1
export CUDASM=$cudasm
export PATH=$ENV_TensorRT_BIN:$CUDA_BIN:$PATH
export LD_LIBRARY_PATH=$ENV_TensorRT_LIB:$ENV_CUDA_LIB:$ENV_cuDNN_LIB:$LD_LIBRARY_PATH

#Part2 Build TensoRT engine.
export ENVTRTDIR=deploy/engine
export ENVTARGETPLUGIN=deploy/dfa_plugin/lib/deformableAttentionAggr.so

export ENV_BACKBONE_ONNX=deploy/onnx/sparse4dbackbone.onnx
export ENV_BACKBONE_ENGINE=${ENVTRTDIR}/sparse4dbackbone.engine

export ENV_HEAD1_ONNX=deploy/onnx/sparse4dhead1st.onnx
export ENV_HEAD1_ENGINE=${ENVTRTDIR}/sparse4dhead1st.engine

export ENV_HEAD2_ONNX=deploy/onnx/sparse4dhead2nd.onnx
export ENV_HEAD2_ENGINE=${ENVTRTDIR}/sparse4dhead2nd.engine

echo "===================================================================================================================="
echo "||  Config Environment Below:"
echo "||  TensorRT LIB \t: $ENV_TensorRT_LIB"
echo "||  TensorRT INC \t: $ENV_TensorRT_INC"
echo "||  TensorRT BIN \t: $ENV_TensorRT_BIN"
echo "||  CUDA_LIB \t: $ENV_CUDA_LIB"
echo "||  CUDA_INC \t: $ENV_CUDA_INC"
echo "||  CUDA_BIN \t: $ENV_CUDA_BIN"
echo "||  CUDNN_LIB \t: $ENV_cuDNN_LIB"
echo "||  CUDASM\t: sm_$cudasm"
echo "||  ENVTRTDIR\t: $ENVTRTDIR"
echo "||  ENVTARGETPLUGIN\t: $ENVTARGETPLUGIN"
echo "||  ENV_BACKBONE_ONNX\t: $ENV_BACKBONE_ONNX"
echo "||  ENV_BACKBONE_ENGINE\t: $ENV_BACKBONE_ENGINE"
echo "||  ENV_HEAD1_ONNX\t: $ENV_HEAD1_ONNX"
echo "||  ENV_HEAD1_ENGINE\t: $ENV_HEAD1_ENGINE"
echo "||  ENV_HEAD2_ONNX\t: $ENV_HEAD2_ONNX"
echo "||  ENV_HEAD2_ENGINE\t: $ENV_HEAD2_ENGINE"
echo "===================================================================================================================="
echo "[INFO] Config Env Done, Please Check EnvPrintOut Above!"

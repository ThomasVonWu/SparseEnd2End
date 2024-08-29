# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

################***EnvVersion-1***#################
###     LinuxPlatform:            x86_64                                                        ###
###     TensorRT :                    8.5.1.7                                                          ###
###     CUDA :                             11.6                                                            ###
###     cuDNN:                          8.6.0.163                                                    ###
###     CUDA capability :        sm_86                                                         ###
################***EnvVersion-1***#################

EnvVersion=1
if [ $EnvVersion = 1 ]; then
    export ENV_TensorRT_LIB=/mnt/env/tensorrt/TensorRT-8.5.1.7/lib
    export ENV_TensorRT_INC=/mnt/env/tensorrt/TensorRT-8.5.1.7/include
    export ENV_TensorRT_BIN=/mnt/env/tensorrt/TensorRT-8.5.1.7/bin
    export ENV_CUDA_LIB=/usr/local/cuda-11.6/lib64
    export ENV_CUDA_INC=/usr/local/cuda-11.6/include
    export ENV_CUDA_BIN=/usr/local/cuda-11.6/bin
    export ENV_cuDNN_LIB=/mnt/env/tensorrt/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib
else
    export ENV_TensorRT_LIB=/mnt/data/env_cfg/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8/TensorRT-8.6.1.6/lib
    export ENV_TensorRT_INC=/mnt/data/env_cfg/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8/TensorRT-8.6.1.6/include
    export ENV_TensorRT_BIN=/mnt/data/env_cfg/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8/TensorRT-8.6.1.6/bin
    export ENV_CUDA_LIB=/usr/local/cuda-11.6/lib64
    export ENV_CUDA_INC=/usr/local/cuda-11.6/include
    export ENV_CUDA_BIN=/usr/local/cuda-11.6/bin
    export ENV_cuDNN_LIB=/mnt/env/tensorrt/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib
fi

if [ ! -f "${ENV_TensorRT_BIN}/trtexec" ]; then
    echo "[ERROR] Failed to Find ${ENV_TensorRT_BIN}/trtexec !"
    return
fi

if [ ! -f "${ENV_CUDA_BIN}/nvcc" ]; then
    echo "[ERROR] Failed to Find ${ENV_CUDA_BIN}/nvcc !"
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
export LD_LIBRARY_PATH=$ENV_TensorRT_LIB:$ENV_CUDA_LIB:$ENVcuDNN_LIB:$ENV_cuDNN_LIB:$LD_LIBRARY_PATH

# Part2 Build op:op_deformableAttentionAggr shared libray, for make .
export ENVBUILDDIR=build                              # relative path for saving make products
export ENVTARGETPLUGIN=lib/deformableAttentionAggr.so # relative path for saving dfa shared library

# Part3 Build TensorRT and test consistency: op_deformableAttentionAggr pytorch->onnx->engine.
export ENVONNX=deploy/dfa_plugin/onnx/deformableAttentionAggr.onnx # absolute path for saving dfa onnx
export ENVTRTDIR=deploy/dfa_plugin/engine                          # absolute path for saving dfa engine convertion logs
export ENVEINGINENAME=$ENVTRTDIR/deformableAttentionAggr.engine    # absolute path for saving dfa engine

echo "===================================================================================================================="
echo "||  Config Environment Below:"
echo "||  TensorRT LIB\t: $ENV_TensorRT_LIB"
echo "||  TensorRT INC\t:  $ENV_TensorRT_INC"
echo "||  TensorRT BIN\t: $ENV_TensorRT_BIN"
echo "||  CUDA_LIB\t: $ENV_CUDA_LIB"
echo "||  CUDA_ INC\t: $ENV_CUDA_INC"
echo "||  CUDA_BIN\t: $ENV_CUDA_BIN"
echo "||  CUDNN_LIB\t: $ENV_cuDNN_LIB"
echo "||  CUDASM\t: sm_$cudasm"
echo "||  ENVBUILDDIR\t: $ENVBUILDDIR"
echo "||  ENVTARGETPLUGIN\t: $ENVTARGETPLUGIN"
echo "||  ENVONNX\t: $ENVONNX"
echo "||  ENVEINGINENAME\t: $ENVEINGINENAME"
echo "||  ENVTRTDIR\t: $ENVTRTDIR"
echo "===================================================================================================================="
echo "[INFO] Config Env Done, Please Check EnvPrintOut Above!"

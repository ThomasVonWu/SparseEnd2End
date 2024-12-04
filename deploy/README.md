# Deploy Sparse4D Pipeline In LocalWorkStation

## STEP1. Export Onnx
```bash
cd /path/to/SparseEnd2End
python deploy/export_backbone_onnx.py --cfg /path/to/cfg --ckpt /path/to/ckpt
python deploy/export_head_onnx.py --cfg /path/to/cfg --ckpt /path/to/ckpt
```
onnx will save in deploy/onnxlog like below:  
>deploy/onnx  
>├── export_backbone_onnx.log  
>├── export_head_onnx.log  
>├── sparse4dbackbone.onnx  
>└── sparse4dhead1st.onnx  
>├── sparse4dhead2nd.onnx  

## STEP2. Compile Custom operator: deformableAttentionAggrPlugin.so
Firstly, you need to set env for youself in 01_setEnv.sh, then run below:
```bash
. deploy/dfa_plugin/tools/01_setEnv.sh
```
env setting likes below:
```bash
====================================================================================================================
||  Config Environment Below:
||  TensorRT LIB        : /mnt/env/tensorrt/TensorRT-8.5.1.7/lib
||  TensorRT INC        :  /mnt/env/tensorrt/TensorRT-8.5.1.7/include
||  TensorRT BIN        : /mnt/env/tensorrt/TensorRT-8.5.1.7/bin
||  CUDA_LIB    : /usr/local/cuda-11.6/lib64
||  CUDA_ INC   : /usr/local/cuda-11.6/include
||  CUDA_BIN    : /usr/local/cuda-11.6/bin
||  CUDNN_LIB   : /mnt/env/tensorrt/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib
||  CUDASM      : sm_86
||  ENVBUILDDIR : build
||  ENVTARGETPLUGIN     : lib/deformableAttentionAggr.so
||  ENVONNX     : deploy/dfa_plugin/onnx/deformableAttentionAggr.onnx
||  ENVEINGINENAME      : deploy/dfa_plugin/engine/deformableAttentionAggr.engine
||  ENVTRTDIR   : deploy/dfa_plugin/engine
====================================================================================================================
[INFO] Config Env Done, Please Check EnvPrintOut Above!
```
then you need to export share library:
```bash
cd deploy/dfa_plugin
make -j8
```
make log likes below:  
```bash
1-Finish Compile CUDA Make Policy build/deformableAttentionAggr.cu.mk
2-Finish Compile CXX Make Policy build/deformableAttentionAggrPlugin.cpp.mk
make lib/deformableAttentionAggr.so
make[1]: Entering directory '/mnt/data/end2endlocal/tmp/SparseEnd2End/deploy/dfa_plugin'
3-Finish Compile CXX Objects : build/deformableAttentionAggrPlugin.cpp.o
4-Finish Compile CUDA Objects build/deformableAttentionAggr.cu.o
5-Finish Compile Target : lib/deformableAttentionAggr.so!
make[1]: Leaving directory '/mnt/data/end2endlocal/tmp/SparseEnd2End/deploy/dfa_plugin'
```

## STEP3. BUILD Sparse4D Engine
Firstly, you need to set env for youself in set_env.sh, then run below:
```bash
cd -
. deploy/tools/set_env.sh
cd deploy
bash build_sparse4d_engine.sh
```
trt log likes below:
>deploy/engine  
>├── build_backbone.engine  
>├── build_head1st.engine  
>├── build_head2nd.engine  
>├── build_backbone.log  
>├── build_head2.log  
>├── build_head1.log  
>├── buildLayerInfo_backbone.json  
>├── buildLayerInfo_head1.json  
>├── buildLayerInfo_head2.json  
>├── buildOutput_backbone.json  
>├── buildOutput_head1.json  
>├── buildOutput_head2.json  
>├── buildProfile_backbone.json  
>├── buildProfile_head1.json  
>└── buildProfile_head2.json  
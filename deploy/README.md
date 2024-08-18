# Deploy Sparse4D Pipeline In LocalWorkStation

## STEP1. Export Onnx
```bash
cd /path/to/e2eod_trackformer
python deploy/nusc_export_onnx/export_backbone_onnx.py --cfg /path/to/cfg --ckpt /path/to/ckpt
python deploy/nusc_export_onnx/export_head_onnx.py --cfg /path/to/cfg --ckpt /path/to/ckpt
```
onnx will save in deploy/onnxlog like below:  
>deploy/onnxlog  
>├── 1st_frame_sparse4dhead.onnx  
>├── export_backbone_onnx.log  
>├── export_head_onnx.log  
>├── sparse4dbackbone.onnx  
>└── sparse4dhead.onnx  

## STEP2. Compile Custom operator: deformableAttentionAggrPlugin.so
You need to first set env for youself in 01_setEnv.sh, then run below:
```bash
cd deploy/dfa_plugin
. tools/01_setEnv.sh
```
env setting likes below:

then you need to export sharelibrary:
```bash
make -j8
```
make log likes below:  
```bash
1-Finish Compile CUDA Make Policy build/deformableAttentionAggr.cu.mk
2-Finish Compile CXX Make Policy build/deformableAttentionAggrPlugin.cpp.mk
make lib/deformableAttentionAggr.so
make[1]: Entering directory '/mnt/data/end2endlocal/tmp/e2eod_trackformer/deploy/dfa_plugin'
3-Finish Compile CXX Objects : build/deformableAttentionAggrPlugin.cpp.o
4-Finish Compile CUDA Objects build/deformableAttentionAggr.cu.o
5-Finish Compile Target : lib/deformableAttentionAggr.so!
make[1]: Leaving directory '/mnt/data/end2endlocal/tmp/e2eod_trackformer/deploy/dfa_plugin'
```

## STEP3. BUILD Sparse4D Engine
You need to first set env for youself in set_env.sh, then run below:
```bash
cd ..
. tools/set_env.sh
bash build_sparse4d_engine.sh
```
trt log likes below:
>trtlog  
>├── build_backbone.log  
>├── build_backbone.engine  
>├── build_head1.log  
>├── build_head1.engine  
>├── build_head2.log  
>├── build_head2.engine  
>├── buildLayerInfo_backbone.json  
>├── buildLayerInfo_head1.json  
>├── buildLayerInfo_head2.json  
>├── buildOutput_backbone.json  
>├── buildOutput_head1.json  
>├── buildOutput_head2.json  
>├── buildProfile_backbone.json  
>├── buildProfile_head1.json  
>└── buildProfile_head2.json  
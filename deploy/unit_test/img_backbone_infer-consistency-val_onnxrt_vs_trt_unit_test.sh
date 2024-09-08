polygraphy run deploy/onnx/sparse4dbackbone.onnx \
    --onnxrt --trt \
    --onnx-outputs mark all \
    --trt-outputs mark all \
    --input-shapes 'img:[1,6,3,256,704]' \
    --atol 1e-4 --rtol 1e-4 \
    --verbose \
    >sparse4d_imgbackbone_consistency_val_onnxrt_vs_trt.log 2>&1

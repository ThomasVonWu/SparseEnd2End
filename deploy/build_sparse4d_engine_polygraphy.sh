ENV_HEAD1_ENGINE="deploy/engine/sparse4dhead1st_polygraphy.engine"
echo "STEP2: build 1st frame sparse4dhead engine -> saving in ${ENV_HEAD1_ENGINE}......"

polygraphy run deploy/onnx/sparse4dhead1st.onnx \
    --trt \
    --verbose \
    --validate \
    --plugins deploy/dfa_plugin/lib/deformableAttentionAggr.so \
    --save-engine ${ENV_HEAD1_ENGINE} \
    >deploy/engine/build_head1_polygraphy.log 2>&1

ENV_DFA_PLUGIN="deploy/dfa_plugin/lib/deformableAttentionAggr.so"

ENV_HEAD1_ENGINE="deploy/engine/sparse4dhead1st_polygraphy.engine"
ENV_HEAD1_ONNX="deploy/onnx/sparse4dhead1st.onnx"
echo "STEP2: Build 1st frame sparse4dhead engine -> saving in ${ENV_HEAD1_ENGINE}......"
polygraphy run ${ENV_HEAD1_ONNX} \
    --trt \
    --verbose \
    --validate \
    --plugins ${ENV_DFA_PLUGIN} \
    --save-engine ${ENV_HEAD1_ENGINE} \
    >deploy/engine/build_head1_polygraphy.log 2>&1

ENV_HEAD2_ENGINE="deploy/engine/sparse4dhead2nd_polygraphy.engine"
ENV_HEAD2_ONNX="deploy/onnx/sparse4dhead2nd.onnx"
echo "STEP3: Build >1 frame sparse4dhead engine -> saving in ${ENV_HEAD2_ENGINE}......"
polygraphy run ${ENV_HEAD2_ONNX} \
    --trt \
    --verbose \
    --validate \
    --plugins ${ENV_DFA_PLUGIN} \
    --save-engine ${ENV_HEAD2_ENGINE} \
    >deploy/engine/build_head2_polygraphy.log 2>&1

#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

if [ ! -d "${ENVTRTLOGSDIR}" ]; then
    mkdir -p "${ENVTRTLOGSDIR}"
fi

${ENV_TensorRT_BIN}/trtexec --onnx=${ENVONNX} \
    --plugins=deploy/dfa_plugin/$ENVTARGETPLUGIN \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENVEINGINENAME} \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTLOGSDIR}/buildOutput.json \
    --exportProfile=${ENVTRTLOGSDIR}/buildProfile.json \
    --exportLayerInfo=${ENVTRTLOGSDIR}/buildLayerInfo.json \
    --profilingVerbosity=detailed \
    >${ENVTRTLOGSDIR}/build.log 2>&1

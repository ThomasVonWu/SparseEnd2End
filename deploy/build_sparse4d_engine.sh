#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

if [ ! -d "${ENVTRTLOGSDIR}" ]; then
    mkdir -p "${ENVTRTLOGSDIR}"
fi

echo "STEP1: build sparse4dbackbone engine -> saving in ${ENV_BACKBONE_ENGINE}..."
# STEP1: build sparse4dbackbone engine
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_BACKBONE_ONNX} \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_BACKBONE_ENGINE} \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTLOGSDIR}/buildOutput_backbone.json --exportProfile=${ENVTRTLOGSDIR}/buildProfile_backbone.json \
    --exportLayerInfo=${ENVTRTLOGSDIR}/buildLayerInfo_backbone.json \
    --profilingVerbosity=detailed \
    >${ENVTRTLOGSDIR}/build_backbone.log 2>&1

echo "STEP2: build 1st frame sparse4dhead engine -> saving in ${ENV_HEAD1_ENGINE}..."
sleep 2s
# STEP2: build 1st frame sparse4dhead engine
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD1_ONNX} \
    --plugins=$ENVTARGETPLUGIN \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_HEAD1_ENGINE} \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTLOGSDIR}/buildOutput_head1.json --exportProfile=${ENVTRTLOGSDIR}/buildProfile_head1.json \
    --exportLayerInfo=${ENVTRTLOGSDIR}/buildLayerInfo_head1.json \
    --profilingVerbosity=detailed \
    >${ENVTRTLOGSDIR}/build_head1.log 2>&1

echo "STEP3: build frame > 2 sparse4dhead engine -> saving in ${ENV_HEAD2_ENGINE}..."
sleep 2s
# STEP3: build frame > 2 sparse4dhead engine
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD2_ONNX} \
    --plugins=$ENVTARGETPLUGIN \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_HEAD2_ENGINE} \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTLOGSDIR}/buildOutput_head2.json --exportProfile=${ENVTRTLOGSDIR}/buildProfile_head2.json \
    --exportLayerInfo=${ENVTRTLOGSDIR}/buildLayerInfo_head2.json \
    --profilingVerbosity=detailed \
    >${ENVTRTLOGSDIR}/build_head2.log 2>&1

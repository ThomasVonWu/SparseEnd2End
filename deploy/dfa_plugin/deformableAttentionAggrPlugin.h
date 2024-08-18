// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef DEPLOY_DFA_PLUGIN_DEFORMABLEATTENTIONAGGRPLUGIN_H
#define DEPLOY_DFA_PLUGIN_DEFORMABLEATTENTIONAGGRPLUGIN_H

#include <string>
#include <vector>

#include <NvInfer.h>

#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"

namespace custom
{
static const char* PLUGIN_NAME{"DeformableAttentionAggrPlugin"};
static const char* PLUGIN_VERSION{"1"};

///@brief First define a Plugin Class: Including Implementation of DeformableAttentionAggrPlugin.
class DeformableAttentionAggrPlugin : public nvinfer1::IPluginV2DynamicExt
{
  public:
    DeformableAttentionAggrPlugin() = default;
    ~DeformableAttentionAggrPlugin() = default;

    /// @brief PART1: Custom Plugin Class: DeformableAttentionAggrPlugin -> nvinfer1::IPluginV2DynamicExt Methods
    /*
     * clone()
     * getOutputDimensions()
     * supportsFormatCombination()
     * configurePlugin()
     * getWorkspaceSize()
     * enqueue()
     * attachToContext()
     * detachFromContext()
     */
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex,
                                            const nvinfer1::DimsExprs* inputs,
                                            int32_t nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos,
                                   const nvinfer1::PluginTensorDesc* inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                         int32_t nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out,
                         int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                            int32_t nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs,
                            int32_t nbOutputs) const noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* ionputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override;
    void attachToContext(cudnnContext* contextCudnn,
                         cublasContext* contextCublas,
                         nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;

    /// @brief PART2: Custom Plugin Class: DeformableAttentionAggrPlugin -> nvinfer1::IPluginV2Ext Methods
    /*
     * getOutputDataType()
     */
    nvinfer1::DataType getOutputDataType(int32_t index,
                                         nvinfer1::DataType const* inputTypes,
                                         int32_t nbInputs) const noexcept override;

    /// @brief PART3: Custom Plugin Class: DeformableAttentionAggrPlugin -> nvinfer1::IPluginV2 Methods
    /*
     * getPluginType()
     * getPluginVersion()
     * getNbOutputs()
     * initialize()
     * getSerializationSize()
     * serialize()
     * destroy()
     * terminate()
     * setPluginNamespace()
     * getPluginNamespace()
     */
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void terminate() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

  private:
    std::string mNamespace_;
};

/// @brief Second define a PluginCreator Class.
/// @brief PART4 Custom Pluginv1 Creator: DeformableAttentionAggrPluginCreator
/*
 * DeformableAttentionAggrPluginCreator()
 * ~DeformableAttentionAggrPluginCreator()
 * getPluginName()
 * getPluginVersion()
 * getFieldNames()
 * createPlugin()
 * deserializePlugin()
 * setPluginNamespace()
 * getPluginNamespace()
 */
class DeformableAttentionAggrPluginCreator : public nvinfer1::IPluginCreator
{
  public:
    DeformableAttentionAggrPluginCreator();
    ~DeformableAttentionAggrPluginCreator() = default;
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                           const void* serialData,
                                           size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

  private:
    nvinfer1::PluginFieldCollection mFC_;
    std::vector<nvinfer1::PluginField> mAttrs_;
    std::string mNamespace_;
};

}  // namespace custom

#endif  // DEPLOY_DFA_PLUGIN_DEFORMABLEATTENTIONAGGRPLUGIN_H

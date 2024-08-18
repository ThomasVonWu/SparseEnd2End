// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <cstdio>

#include <cuda_runtime.h>
__device__ float thomas_bilinear_sampling(const float*& bottom_data,
                                         const int32_t& height,
                                         const int32_t& width,
                                         const int32_t& num_embeds,
                                         const float& h_im,
                                         const float& w_im,
                                         const int32_t& base_ptr)
{
    const int32_t h_low = floorf(h_im);
    const int32_t w_low = floorf(w_im);
    const int32_t h_high = h_low + 1;
    const int32_t w_high = w_low + 1;

    const float lh = h_im - h_low;
    const float lw = w_im - w_low;
    const float hh = 1 - lh, hw = 1 - lw;

    const int32_t w_stride = num_embeds;
    const int32_t h_stride = width * w_stride;
    const int32_t h_low_ptr_offset = h_low * h_stride;
    const int32_t h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int32_t w_low_ptr_offset = w_low * w_stride;
    const int32_t w_high_ptr_offset = w_low_ptr_offset + w_stride;

    float v1 = 0;
    if (h_low >= 0 && w_low >= 0)
    {
        const int32_t ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
    }
    float v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
    {
        const int32_t ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
    }
    float v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
    {
        const int32_t ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
    }
    float v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        const int32_t ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
    }

    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

__global__ void thomas_deformable_aggregation_kernel(
    const int32_t num_kernels,         // batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    float* output,                     // batch_size * num_anchors * num_embeds
    const float* mc_ms_feat,           // batch_size * num_feat * num_embeds
    const int32_t* spatial_shape,      // num_cams * num_scale * 2
    const int32_t* scale_start_index,  /// num_cams * num_scale
    const float* sample_location,      /// batch_size * num_anchors * num_pts * num_cams * 2
    const float* weights,              /// batch_size * num_anchors * num_pts * num_cams * num_scale * num_groups
    int32_t batch_size,
    int32_t num_cams,
    int32_t num_feat,
    int32_t num_embeds,
    int32_t num_scale,
    int32_t num_anchors,
    int32_t num_pts,
    int32_t num_groups)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels)
        return;

    const float weight = *(weights + idx / (num_embeds / num_groups));
    const int32_t channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int32_t scale_index = idx % num_scale;
    idx /= num_scale;

    const int32_t cam_index = idx % num_cams;
    idx /= num_cams;
    const int32_t pts_index = idx % num_pts;
    idx /= num_pts;

    int32_t anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int32_t batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int32_t loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1)
        return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1)
        return;

    int32_t cam_scale_index = cam_index * num_scale + scale_index;
    const int32_t value_offset =
        (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    if (value_offset > batch_size * num_feat * num_embeds || value_offset < 0)
        return;

    cam_scale_index = cam_scale_index << 1;
    const int32_t h = spatial_shape[cam_scale_index];
    const int32_t w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    if (h_im > -1 && w_im > -1 && h_im < h && w_im < w)
    {
        atomicAdd(output + anchor_index * num_embeds + channel_index,
                  thomas_bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight);
    }
}

int32_t thomas_deform_attn_cuda_forward(cudaStream_t stream,
                                       const float* value,
                                       const int32_t* spatialShapes,
                                       const int32_t* levelStartIndex,
                                       const float* samplingLoc,
                                       const float* attnWeight,
                                       float* output,
                                       int32_t batch,
                                       int32_t mSpatialSize,
                                       int32_t mChannels,
                                       int32_t mNumCams,
                                       int32_t mNumLevels,
                                       int32_t mNumQuery,
                                       int32_t mNumPoint,
                                       int32_t mNumGroups)
{
    const int32_t num_kernels = batch * mNumCams * mNumLevels * mNumQuery * mNumPoint * mChannels;
    cudaError_t err = cudaSuccess;

    thomas_deformable_aggregation_kernel<<<(int32_t)ceil(((double)num_kernels / 128)), 128, 0, stream>>>(num_kernels,
                                                                                                        output,
                                                                                                        value,
                                                                                                        spatialShapes,
                                                                                                        levelStartIndex,
                                                                                                        samplingLoc,
                                                                                                        attnWeight,
                                                                                                        batch,
                                                                                                        mNumCams,
                                                                                                        mSpatialSize,
                                                                                                        mChannels,
                                                                                                        mNumLevels,
                                                                                                        mNumQuery,
                                                                                                        mNumPoint,
                                                                                                        mNumGroups);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("[DFA-PLUGIN-ERROR] Kernel Launch Failed with Error \"%s : %s\".\n",
               cudaGetErrorString(err),
               cudaGetErrorName(err));
    }

    return 0;
}
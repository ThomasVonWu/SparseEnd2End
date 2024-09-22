// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "common.h"
#include "img_aug_with_bilinearinterpolation_kernel.h"
namespace sparse_end2end {
namespace image {

#define R_MEAN 0.485F
#define G_MEAN 0.456F
#define B_MEAN 0.406F
#define R_STD 0.229F
#define G_STD 0.224F
#define B_STD 0.225F

#define DIVUP(a, b) ((a % b != 0) ? (a / b + 1) : (a / b))

// image prepocess  CUDA kernel with output dtype: fp32.
__global__ void imgAugKernel(const std::uint8_t* input,
                             const std::uint32_t n,
                             const std::uint32_t c,
                             const std::uint32_t h,
                             const std::uint32_t w,
                             const std::uint32_t target_h,
                             const std::uint32_t target_w,
                             const std::uint32_t crop_h,
                             const std::uint32_t crop_w,
                             const float resize_ratio,
                             float* output) {
  const std::int32_t cam_id = blockIdx.x;
  const std::int32_t dst_y = blockIdx.y * blockDim.x + threadIdx.x;
  const std::int32_t dst_x = blockIdx.z * blockDim.y + threadIdx.y;

  if (dst_y >= target_h || dst_x >= target_w) {
    return;
  }

  const float resize_ratio_x = static_cast<float>(w) / static_cast<float>(std::floor(w * resize_ratio));
  const float resize_ratio_y = static_cast<float>(h) / static_cast<float>(std::floor(h * resize_ratio));

  const float src_x = (dst_x + crop_w + 0.5F) * resize_ratio_x - 0.5F;
  const float src_y = (dst_y + crop_h + 0.5F) * resize_ratio_y - 0.5F;

  std::uint32_t low_x = std::floor(src_x);
  std::uint32_t low_y = std::floor(src_y);

  std::uint32_t high_x = min(low_x + 1U, w - 1U);
  std::uint32_t high_y = min(low_y + 1U, h - 1U);

  low_x = max(0U, low_x);
  low_y = max(0U, low_y);

  const float ly = src_y - low_y;
  const float lx = src_x - low_x;
  const float hy = 1.0F - ly;
  const float hx = 1.0F - lx;

  const float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  const float value1_r = float(input[cam_id * (c * h * w) + 0 * (h * w) + low_y * w + low_x]);
  const float value1_g = float(input[cam_id * (c * h * w) + 1 * (h * w) + low_y * w + low_x]);
  const float value1_b = float(input[cam_id * (c * h * w) + 2 * (h * w) + low_y * w + low_x]);

  const float value2_r = float(input[cam_id * (c * h * w) + 0 * (h * w) + low_y * w + high_x]);
  const float value2_g = float(input[cam_id * (c * h * w) + 1 * (h * w) + low_y * w + high_x]);
  const float value2_b = float(input[cam_id * (c * h * w) + 2 * (h * w) + low_y * w + high_x]);

  const float value3_r = float(input[cam_id * (c * h * w) + 0 * (h * w) + high_y * w + low_x]);
  const float value3_g = float(input[cam_id * (c * h * w) + 1 * (h * w) + high_y * w + low_x]);
  const float value3_b = float(input[cam_id * (c * h * w) + 2 * (h * w) + high_y * w + low_x]);

  const float value4_r = float(input[cam_id * (c * h * w) + 0 * (h * w) + high_y * w + high_x]);
  const float value4_g = float(input[cam_id * (c * h * w) + 1 * (h * w) + high_y * w + high_x]);
  const float value4_b = float(input[cam_id * (c * h * w) + 2 * (h * w) + high_y * w + high_x]);

  float r_value = value1_r * w1 + value2_r * w2 + value3_r * w3 + value4_r * w4;
  float g_value = value1_g * w1 + value2_g * w2 + value3_g * w3 + value4_g * w4;
  float b_value = value1_b * w1 + value2_b * w2 + value3_b * w3 + value4_b * w4;

  r_value = r_value / 255.0F;
  g_value = g_value / 255.0F;
  b_value = b_value / 255.0F;

  r_value = (r_value - R_MEAN) / R_STD;
  g_value = (g_value - G_MEAN) / G_STD;
  b_value = (b_value - B_MEAN) / B_STD;

  output[cam_id * c * target_h * target_w + 0U * target_h * target_w + dst_y * target_w + dst_x] = r_value;
  output[cam_id * c * target_h * target_w + 1U * target_h * target_w + dst_y * target_w + dst_x] = g_value;
  output[cam_id * c * target_h * target_w + 2U * target_h * target_w + dst_y * target_w + dst_x] = b_value;
}

// image prepocess  CUDA kernel with output dtype: fp16.
__global__ void imgAugKernel(const std::uint8_t* input,
                             const std::uint32_t n,
                             const std::uint32_t c,
                             const std::uint32_t h,
                             const std::uint32_t w,
                             const std::uint32_t target_h,
                             const std::uint32_t target_w,
                             const std::uint32_t crop_h,
                             const std::uint32_t crop_w,
                             const float resize_ratio,
                             half* output) {
  const std::int32_t cam_id = blockIdx.x;
  const std::int32_t dst_y = blockIdx.y * blockDim.x + threadIdx.x;
  const std::int32_t dst_x = blockIdx.z * blockDim.y + threadIdx.y;

  if (dst_y >= target_h || dst_x >= target_w) {
    return;
  }

  const float resize_ratio_x = static_cast<float>(w) / static_cast<float>(std::floor(w * resize_ratio));
  const float resize_ratio_y = static_cast<float>(h) / static_cast<float>(std::floor(h * resize_ratio));

  const float src_x = (dst_x + crop_w + 0.5F) * resize_ratio_x - 0.5F;
  const float src_y = (dst_y + crop_h + 0.5F) * resize_ratio_y - 0.5F;

  std::uint32_t low_x = std::floor(src_x);
  std::uint32_t low_y = std::floor(src_y);

  std::uint32_t high_x = min(low_x + 1U, w - 1U);
  std::uint32_t high_y = min(low_y + 1U, h - 1U);

  low_x = max(0U, low_x);
  low_y = max(0U, low_y);

  const float ly = src_y - low_y;
  const float lx = src_x - low_x;
  const float hy = 1.0F - ly;
  const float hx = 1.0F - lx;

  const float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  const float value1_r = float(input[cam_id * (c * h * w) + 0 * (h * w) + low_y * w + low_x]);
  const float value1_g = float(input[cam_id * (c * h * w) + 1 * (h * w) + low_y * w + low_x]);
  const float value1_b = float(input[cam_id * (c * h * w) + 2 * (h * w) + low_y * w + low_x]);

  const float value2_r = float(input[cam_id * (c * h * w) + 0 * (h * w) + low_y * w + high_x]);
  const float value2_g = float(input[cam_id * (c * h * w) + 1 * (h * w) + low_y * w + high_x]);
  const float value2_b = float(input[cam_id * (c * h * w) + 2 * (h * w) + low_y * w + high_x]);

  const float value3_r = float(input[cam_id * (c * h * w) + 0 * (h * w) + high_y * w + low_x]);
  const float value3_g = float(input[cam_id * (c * h * w) + 1 * (h * w) + high_y * w + low_x]);
  const float value3_b = float(input[cam_id * (c * h * w) + 2 * (h * w) + high_y * w + low_x]);

  const float value4_r = float(input[cam_id * (c * h * w) + 0 * (h * w) + high_y * w + high_x]);
  const float value4_g = float(input[cam_id * (c * h * w) + 1 * (h * w) + high_y * w + high_x]);
  const float value4_b = float(input[cam_id * (c * h * w) + 2 * (h * w) + high_y * w + high_x]);

  float r_value = value1_r * w1 + value2_r * w2 + value3_r * w3 + value4_r * w4;
  float g_value = value1_g * w1 + value2_g * w2 + value3_g * w3 + value4_g * w4;
  float b_value = value1_b * w1 + value2_b * w2 + value3_b * w3 + value4_b * w4;

  r_value = r_value / 255.0F;
  g_value = g_value / 255.0F;
  b_value = b_value / 255.0F;

  r_value = (r_value - R_MEAN) / R_STD;
  g_value = (g_value - G_MEAN) / G_STD;
  b_value = (b_value - B_MEAN) / B_STD;

  output[cam_id * c * target_h * target_w + 0U * target_h * target_w + dst_y * target_w + dst_x] =
      __float2half(r_value);
  output[cam_id * c * target_h * target_w + 1U * target_h * target_w + dst_y * target_w + dst_x] =
      __float2half(g_value);
  output[cam_id * c * target_h * target_w + 2U * target_h * target_w + dst_y * target_w + dst_x] =
      __float2half(b_value);
}

Status imgPreprocessLauncher(const std::uint8_t* raw_imgs_ptr,
                             const std::uint32_t num_cams,
                             const std::uint32_t raw_img_c,
                             const std::uint32_t raw_img_h,
                             const std::uint32_t raw_img_w,
                             const std::uint32_t target_img_h,
                             const std::uint32_t target_img_w,
                             const std::uint32_t crop_h,
                             const std::uint32_t crop_w,
                             const float resize_ratio,
                             const cudaStream_t& stream,
                             float* output_images_cuda_ptr)

{
  const std::uint32_t thread_num = 32U;
  dim3 blocks_dim_in_each_grid(num_cams, DIVUP(target_img_h, thread_num), DIVUP(target_img_w, thread_num));
  dim3 threads_dim_in_each_block(thread_num, thread_num);

  imgAugKernel<<<blocks_dim_in_each_grid, threads_dim_in_each_block, 0, stream>>>(
      raw_imgs_ptr, num_cams, raw_img_c, raw_img_h, raw_img_w, target_img_h, target_img_w, crop_h, crop_w, resize_ratio,
      output_images_cuda_ptr);

  if (cudaError::cudaSuccess == cudaGetLastError()) {
    return Status::kSuccess;
  } else {
    return Status::kImgPreprocessLaunchErr;
  }
}

Status ImgPreprocessLauncher(const std::uint8_t* raw_imgs_ptr,
                             const std::uint32_t num_cams,
                             const std::uint32_t raw_img_c,
                             const std::uint32_t raw_img_h,
                             const std::uint32_t raw_img_w,
                             const std::uint32_t target_img_h,
                             const std::uint32_t target_img_w,
                             const std::uint32_t crop_h,
                             const std::uint32_t crop_w,
                             const float resize_ratio,
                             const cudaStream_t& stream,
                             half* output_images_cuda_ptr)

{
  const std::uint32_t thread_num = 32U;
  dim3 blocks_dim_in_each_grid(num_cams, DIVUP(target_img_h, thread_num), DIVUP(target_img_w, thread_num));
  dim3 threads_dim_in_each_block(thread_num, thread_num);

  imgAugKernel<<<blocks_dim_in_each_grid, threads_dim_in_each_block, 0, stream>>>(
      raw_imgs_ptr, num_cams, raw_img_c, raw_img_h, raw_img_w, target_img_h, target_img_w, crop_h, crop_w, resize_ratio,
      output_images_cuda_ptr);

  if (cudaError::cudaSuccess == cudaGetLastError()) {
    return Status::kSuccess;
  } else {
    return Status::kImgPreprocessLaunchErr;
  }
}

}  // namespace image
}  // namespace sparse_end2end

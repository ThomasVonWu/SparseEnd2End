// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                                                          \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x) \
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); j += blockDim.y * gridDim.y)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK)
{
    int optimal_block_num = (N + num_threads - 1) / num_threads;
    int max_block_num = 4096;
    return min(optimal_block_num, max_block_num);
}

template <typename T>
__global__ void sigmoid_focal_loss_forward_cuda_kernel(const int nthreads,
                                                       const T* input,
                                                       const int64_t* target,
                                                       const T* weight,
                                                       T* output,
                                                       const T gamma,
                                                       const T alpha,
                                                       const int num_classes)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int n = index / num_classes;
        int c = index % num_classes;

        int64_t t = target[n];
        T flag_p = (t == c);
        T flag_n = (t != c);

        // p = sigmoid(x) = 1. / 1. + expf(-x)
        T p = (T)1. / ((T)1. + expf(-input[index]));

        // (1 - p)**gamma * log(p)
        T term_p = pow(((T)1. - p), gamma) * log(max(p, (T)FLT_MIN));
        // p**gamma * log(1 - p)
        T term_n = pow(p, gamma) * log(max((T)1. - p, (T)FLT_MIN));

        output[index] = (T)0.;
        output[index] += -flag_p * alpha * term_p;
        output[index] += -flag_n * ((T)1. - alpha) * term_n;
        if (weight != NULL)
        {
            output[index] *= weight[t];
        }
    }
}

template <typename T>
__global__ void sigmoid_focal_loss_backward_cuda_kernel(const int nthreads,
                                                        const T* input,
                                                        const int64_t* target,
                                                        const T* weight,
                                                        T* grad_input,
                                                        const T gamma,
                                                        const T alpha,
                                                        const int num_classes)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int n = index / num_classes;
        int c = index % num_classes;

        int64_t t = target[n];
        T flag_p = (t == c);
        T flag_n = (t != c);

        // p = sigmoid(x) = 1. / 1. + expf(-x)
        T p = (T)1. / ((T)1. + exp(-input[index]));

        // (1 - p)**gamma * (1 - p - gamma*p*log(p))
        T term_p = pow(((T)1. - p), gamma) * ((T)1. - p - (gamma * p * log(max(p, (T)FLT_MIN))));
        // p**gamma * (gamma * (1 - p) * log(1 - p) - p)
        T term_n = pow(p, gamma) * (gamma * ((T)1. - p) * log(max((T)1. - p, (T)FLT_MIN)) - p);

        grad_input[index] = (T)0.;
        grad_input[index] += -flag_p * alpha * term_p;
        grad_input[index] += -flag_n * ((T)1. - alpha) * term_n;
        if (weight != NULL)
        {
            grad_input[index] *= weight[t];
        }
    }
}

void SigmoidFocalLossForwardCUDAKernelLauncher(at::Tensor input,
                                               at::Tensor target,
                                               at::Tensor weight,
                                               at::Tensor output,
                                               const float gamma,
                                               const float alpha)
{
    int output_size = output.numel();
    int num_classes = input.size(1);
    AT_ASSERTM(target.max().item<int64_t>() <= (int64_t)num_classes,
               "target label should smaller or equal than num classes");
    at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sigmoid_focal_loss_forward_cuda_kernel", [&] {
        sigmoid_focal_loss_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(output_size,
                                                                        input.data_ptr<scalar_t>(),
                                                                        target.data_ptr<int64_t>(),
                                                                        weight.data_ptr<scalar_t>(),
                                                                        output.data_ptr<scalar_t>(),
                                                                        gamma,
                                                                        alpha,
                                                                        num_classes);
    });

    AT_CUDA_CHECK(cudaGetLastError());
}

void SigmoidFocalLossBackwardCUDAKernelLauncher(at::Tensor input,
                                                at::Tensor target,
                                                at::Tensor weight,
                                                at::Tensor grad_input,
                                                const float gamma,
                                                const float alpha)
{
    int output_size = grad_input.numel();
    int num_classes = input.size(1);

    at::cuda::CUDAGuard device_guard(grad_input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sigmoid_focal_loss_backward_cuda_kernel", [&] {
        sigmoid_focal_loss_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(output_size,
                                                                        input.data_ptr<scalar_t>(),
                                                                        target.data_ptr<int64_t>(),
                                                                        weight.data_ptr<scalar_t>(),
                                                                        grad_input.data_ptr<scalar_t>(),
                                                                        gamma,
                                                                        alpha,
                                                                        num_classes);
    });

    AT_CUDA_CHECK(cudaGetLastError());
}
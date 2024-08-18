// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <torch/torch.h>

void SigmoidFocalLossForwardCUDAKernelLauncher(at::Tensor input,
                                               at::Tensor target,
                                               at::Tensor weight,
                                               at::Tensor output,
                                               const float gamma,
                                               const float alpha);

void SigmoidFocalLossBackwardCUDAKernelLauncher(at::Tensor input,
                                                at::Tensor target,
                                                at::Tensor weight,
                                                at::Tensor grad_input,
                                                const float gamma,
                                                const float alpha);
void sigmoid_focal_loss_forward_cuda(at::Tensor input,
                                     at::Tensor target,
                                     at::Tensor weight,
                                     at::Tensor output,
                                     float gamma,
                                     float alpha)
{
    SigmoidFocalLossForwardCUDAKernelLauncher(input, target, weight, output, gamma, alpha);
}

void sigmoid_focal_loss_backward_cuda(at::Tensor input,
                                      at::Tensor target,
                                      at::Tensor weight,
                                      at::Tensor grad_input,
                                      float gamma,
                                      float alpha)
{
    SigmoidFocalLossBackwardCUDAKernelLauncher(input, target, weight, grad_input, gamma, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sigmoid_focal_loss_forward_cuda",
          &sigmoid_focal_loss_forward_cuda,
          "CUDA Implementation of sigmoid focal loss forward.",
          py::arg("input"),
          py::arg("target"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("gamma"),
          py::arg("alpha"));
    m.def("sigmoid_focal_loss_backward_cuda",
          &sigmoid_focal_loss_backward_cuda,
          "CUDA Implementation of sigmoid focal backward.",
          py::arg("input"),
          py::arg("target"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("gamma"),
          py::arg("alpha"));
}

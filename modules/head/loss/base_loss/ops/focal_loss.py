# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch

from typing import Optional, Union
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from . import e2e_sigmoid_focal_loss_ext


class SigmoidFocalLossFunction(Function):
    @staticmethod
    def symbolic(
        g,
        input: torch.Tensor,
        target: torch.LongTensor,
        gamma: float,
        alpha: float,
        weight: torch.Tensor,
        reduction: str,
    ):
        return g.op(
            "e2e::E2ESigmoidFocalLoss",
            input,
            target,
            gamma_f=gamma,
            alpha_f=alpha,
            weight_f=weight,
            reduction_s=reduction,
        )

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        target: Union[torch.LongTensor, torch.cuda.LongTensor],
        gamma: float = 2.0,
        alpha: float = 0.25,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:

        assert target.dtype == torch.long
        assert input.dim() == 2
        assert target.dim() == 1
        assert input.size(0) == target.size(0)
        if weight is None:
            weight = input.new_empty(0)
        else:
            assert weight.dim() == 1
            assert input.size(1) == weight.size(0)
        ctx.reduction_dict = {"none": 0, "mean": 1, "sum": 2}
        assert reduction in ctx.reduction_dict.keys()

        ctx.gamma = float(gamma)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        output = input.new_zeros(input.size())

        e2e_sigmoid_focal_loss_ext.sigmoid_focal_loss_forward_cuda(
            input, target, weight, output, gamma=ctx.gamma, alpha=ctx.alpha
        )
        if ctx.reduction == ctx.reduction_dict["mean"]:
            output = output.sum() / input.size(0)
        elif ctx.reduction == ctx.reduction_dict["sum"]:
            output = output.sum()
        ctx.save_for_backward(input, target, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        input, target, weight = ctx.saved_tensors

        grad_input = input.new_zeros(input.size())

        e2e_sigmoid_focal_loss_ext.sigmoid_focal_loss_backward_cuda(
            input, target, weight, grad_input, gamma=ctx.gamma, alpha=ctx.alpha
        )

        grad_input *= grad_output
        if ctx.reduction == ctx.reduction_dict["mean"]:
            grad_input /= input.size(0)
        return grad_input, None, None, None, None, None


sigmoid_focal_loss = SigmoidFocalLossFunction.apply

# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import logging
from typing import Optional, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad

from ..runner.fp16_utils import wrap_fp16_model
from .hook import Hook

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass


class OptimizerHook(Hook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """

    def __init__(
        self, grad_clip: Optional[dict] = None, detect_anomalous_params: bool = False
    ):
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs["loss"], runner)
        runner.outputs["loss"].backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update(
                    {"grad_norm": float(grad_norm)}, runner.outputs["num_samples"]
                )
        runner.optimizer.step()

    def detect_anomalous_parameters(self, loss: Tensor, runner) -> None:
        logger = runner.logger
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, "variable"):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in runner.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f"{n} with shape {p.size()} is not "
                    f"in the computational graph \n",
                )


class Fp16OptimizerHook(OptimizerHook):
    """FP16 optimizer hook (using PyTorch's implementation).

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
    to take care of the optimization procedure.

    Args:
        loss_scale (float | str | dict): Scale factor configuration.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of GradScalar.
            Defaults to 512. For Pytorch >= 1.6, E2E uses official
            implementation of GradScaler. If you use a dict version of
            loss_scale to create GradScaler, please refer to:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
            for the parameters.

    Examples:
        >>> loss_scale = dict(
        ...     init_scale=65536.0,
        ...     growth_factor=2.0,
        ...     backoff_factor=0.5,
        ...     growth_interval=2000
        ... )
        >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
    """

    def __init__(
        self,
        grad_clip: Optional[dict] = None,
        coalesce: bool = True,
        bucket_size_mb: int = -1,
        loss_scale: Union[float, str, dict] = 512.0,
        distributed: bool = True,
    ):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        self._scale_update_param = None
        if loss_scale == "dynamic":
            self.loss_scaler = GradScaler()
        elif isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.loss_scaler = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.loss_scaler = GradScaler(**loss_scale)
        else:
            raise ValueError(
                "loss_scale must be of type float, dict, or "
                f'"dynamic", got {loss_scale}'
            )

    def before_run(self, runner) -> None:
        """Preparing steps before Mixed Precision Training."""
        # wrap model mode to fp16
        wrap_fp16_model(runner.model)
        # resume from state dict
        if "fp16" in runner.meta and "loss_scaler" in runner.meta["fp16"]:
            scaler_state_dict = runner.meta["fp16"]["loss_scaler"]
            self.loss_scaler.load_state_dict(scaler_state_dict)

    def after_train_iter(self, runner) -> None:
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer to
        https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients.
        3. Unscale the optimizer’s gradient tensors.
        4. Call optimizer.step() and update scale factor.
        5. Save loss_scaler state_dict for resume purpose.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()

        self.loss_scaler.scale(runner.outputs["loss"]).backward()
        self.loss_scaler.unscale_(runner.optimizer)
        # grad clip
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update(
                    {"grad_norm": float(grad_norm)}, runner.outputs["num_samples"]
                )
        # backward and update scaler
        self.loss_scaler.step(runner.optimizer)
        self.loss_scaler.update(self._scale_update_param)

        # save state_dict of loss_scaler
        runner.meta.setdefault("fp16", {})[
            "loss_scaler"
        ] = self.loss_scaler.state_dict()

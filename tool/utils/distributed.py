# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
from typing import Any, List, Tuple

import torch
from torch.nn.parallel.distributed import DistributedDataParallel, _find_tensors
from dataset.utils.scatter_gather import ScatterInputs, scatter_kwargs

__all__ = [
    "E2EDistributedDataParallel",
]


class E2EDistributedDataParallel(DistributedDataParallel):
    """The DDP module that supports DataContainer.

    E2EDDP has two main differences with PyTorch DDP:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data.
    - It implement two APIs ``train_step()`` and ``val_step()``.
    """

    def to_kwargs(
        self, inputs: ScatterInputs, kwargs: ScatterInputs, device_id: int
    ) -> Tuple[tuple, tuple]:
        # Use `self.to_kwargs` instead of `self.scatter` in pytorch1.8
        # to move all tensors to device_id
        return scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)

    def scatter(
        self, inputs: ScatterInputs, kwargs: ScatterInputs, device_ids: List[int]
    ) -> Tuple[tuple, tuple]:
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def train_step(self, *inputs, **kwargs):
        """train_step() API for module wrapped by DistributedDataParallel.

        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.train_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """

        if self._check_sync_bufs_pre_fwd():
            self._sync_buffers()

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.train_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[: len(inputs)], inputs, kwargs
                )
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.train_step(*inputs, **kwargs)

        if self._check_sync_bufs_post_fwd():
            self._sync_buffers()

        if (
            torch.is_grad_enabled()
            and getattr(self, "require_backward_grad_sync", False)
            and self.require_backward_grad_sync
        ):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False
        return output

    def val_step(self, *inputs, **kwargs):
        """val_step() API for module wrapped by DistributedDataParallel.

        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.val_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """
        # In PyTorch >= 1.7, ``reducer._rebuild_buckets()`` is moved from the
        # end of backward to the beginning of forward.
        if self._check_sync_bufs_pre_fwd():
            self._sync_buffers()

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.val_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[: len(inputs)], inputs, kwargs
                )
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.val_step(*inputs, **kwargs)

        if self._check_sync_bufs_post_fwd():
            self._sync_buffers()

        if (
            torch.is_grad_enabled()
            and getattr(self, "require_backward_grad_sync", False)
            and self.require_backward_grad_sync
        ):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False
        return output

    def _run_ddp_forward(self, *inputs, **kwargs) -> Any:
        """Processes inputs and runs ``self.module.forward``.

        Pytorch 1.12.0 performs ``self.module.forward`` in ``_run_ddp_forward``
        and deprecates using ``DistributedDataParallel.to_kwargs`` to
        process inputs, which leads to inputs cannot be processed by
        :meth:`E2EDistributedDataParallel.to_kwargs` anymore. Therefore,
        ``E2EDistributedDataParallel`` overrides this method to call
        :meth:`to_kwargs` explicitly.

        Returns:
            Any: Forward result of :attr:`module`.
        """
        module_to_run = (
            self._replicated_tensor_module
            if self._use_replicated_tensor_module
            else self.module
        )

        if self.device_ids:
            inputs, kwargs = self.to_kwargs(  # type: ignore
                inputs, kwargs, self.device_ids[0]
            )
            return module_to_run(*inputs[0], **kwargs[0])  # type: ignore
        else:
            return module_to_run(*inputs, **kwargs)

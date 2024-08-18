# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import time
import warnings
import os.path as osp
from typing import Callable, Dict, List, Optional, Tuple, Union, no_type_check

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .base_runner import BaseRunner
from .checkpoint import save_checkpoint
from ..hook.iter_timer import IterTimerHook

from getpass import getuser
from socket import gethostname

__all__ = ["IterBasedRunner"]


class IterBasedRunner(BaseRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook("before_train_iter")
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError("model.train_step() must return a dict")
        if "log_vars" in outputs:
            self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
        self.outputs = outputs
        self.call_hook("after_train_iter")
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook("before_val_iter")
        outputs = self.model.val_step(data_batch, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError("model.val_step() must return a dict")
        if "log_vars" in outputs:
            self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
        self.outputs = outputs
        self.call_hook("after_val_iter")
        del self.data_batch
        self._inner_iter += 1

    def run(
        self,
        data_loaders: List[DataLoader],
        workflow: List[Tuple[str, int]],
        **kwargs,
    ) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert len(data_loaders) == len(workflow)
        assert (
            self._max_iters is not None
        ), "max_iters must be specified during instantiation"

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info(
            "Start running, host: %s, work_dir: %s", get_host_info(), work_dir
        )
        # self.logger.info(
        #     "Hooks will be executed in the following order:\n%s", self.get_hook_info()
        # )
        self.logger.info("workflow: %s, max: %d iters", workflow, self._max_iters)

        self.call_hook("before_run")
        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook("before_epoch")

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == "train" and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook("after_epoch")
        self.call_hook("after_run")

    @no_type_check
    def resume(
        self,
        checkpoint: str,
        resume_optimizer: bool = True,
        map_location: Union[str, Callable] = "default",
    ) -> None:
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
            map_location (str, optional): Same as :func:`torch.load`.
                Default to 'default'.
        """
        if map_location == "default":
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=lambda storage, loc: storage.cuda(device_id)
            )
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)

        self._epoch = checkpoint["meta"]["epoch"]
        self._iter = checkpoint["meta"]["iter"]
        self._inner_iter = checkpoint["meta"]["iter"]
        if "optimizer" in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(checkpoint["optimizer"][k])
            else:
                raise TypeError(
                    "Optimizer should be dict or torch.optim.Optimizer "
                    f"but got {type(self.optimizer)}"
                )

        self.logger.info(f"resumed from epoch: {self.epoch}, iter {self.iter}")

    def save_checkpoint(  # type: ignore
        self,
        out_dir: str,
        filename_tmpl: str = "iter_{}.pth",
        meta: Optional[Dict] = None,
        save_optimizer: bool = True,
        create_symlink: bool = True,
    ) -> None:
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(f"meta should be a dict or None, but got {type(meta)}")
        if self.meta is not None:
            meta.update(self.meta)
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, "latest.pth")
            if osp.lexists(dst_file):
                os.remove(dst_file)
            os.symlink(filename, dst_file)

    def register_training_hooks(
        self,
        lr_config,
        optimizer_config=None,
        checkpoint_config=None,
        log_config=None,
    ):
        """Register default hooks for iter-based training.

        Checkpoint hook, optimizer stepper hook and logger hooks will be set to
        `by_epoch=False` by default.

        Default hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | Fp16OptimizerHook    | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointHook       | NORMAL (50)             |
        +----------------------+-------------------------+
        | EvalHook             | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | TextLoggerHook       | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | TensorboardLoggerHook| VERY_LOW (90)           |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        if checkpoint_config is not None:
            checkpoint_config.setdefault("by_epoch", False)  # type: ignore
        if lr_config is not None:
            lr_config.setdefault("by_epoch", False)  # type: ignore
        if log_config is not None:
            for info in log_config["hooks"]:
                info.setdefault("by_epoch", False)
        super().register_training_hooks(
            lr_config=lr_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            log_config=log_config,
            timer_config=IterTimerHook(),
        )


def get_host_info() -> str:
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ""
    try:
        host = f"{getuser()}@{gethostname()}"
    except Exception as e:
        warnings.warn(f"Host or user not found: {str(e)}")
    finally:
        return host


class IterLoader:
    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch"):
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)

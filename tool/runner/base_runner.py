# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, no_type_check

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import time
from ..utils.dist_utils import get_dist_info

from .priority import *
from .checkpoint import load_checkpoint
from .log_buffer import LogBuffer
from ..hook import *


class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for PyTorch.

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``val()``
    - ``save_checkpoint()``

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        optimizer (dict or :obj:`torch.optim.Optimizer`): It can be either an
            optimizer (in most cases) or a dict of optimizers (in models that
            requires more than one optimizer, e.g., GAN).
        work_dir (str, optional): The working directory to save checkpoints
            and logs. Defaults to None.
        logger (:obj:`logging.Logger`): Logger used during training.
             Defaults to None. (The default value is just for backward
             compatibility)
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
        max_epochs (int, optional): Total training epochs.
        max_iters (int, optional): Total training iterations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        work_dir: str,
        optimizer: Union[Dict, torch.optim.Optimizer, None] = None,
        logger: Optional[logging.Logger] = None,
        meta: Optional[Dict] = None,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> None:

        assert hasattr(model, "train_step")

        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta
        self.work_dir = os.path.abspath(work_dir)

        # get model name from the model class
        if hasattr(self.model, "module"):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.mode: Optional[str] = None
        self._hooks: List[Hook] = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        if max_epochs is not None and max_iters is not None:
            raise ValueError("Only one of `max_epochs` or `max_iters` can be set.")

        self._max_epochs = max_epochs
        self._max_iters = max_iters
        # TODO: Redesign LogBuffer, it is not flexible and elegant enough
        self.log_buffer = LogBuffer()

    @property
    def model_name(self) -> str:
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self) -> int:
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self) -> int:
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self) -> List[Hook]:
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self) -> int:
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self) -> int:
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self) -> int:
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def run(
        self, data_loaders: List[DataLoader], workflow: List[Tuple[str, int]], **kwargs
    ) -> Any:
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        out_dir: str,
        filename_tmpl: str,
        save_optimizer: bool = True,
        meta: Optional[Dict] = None,
        create_symlink: bool = True,
    ) -> None:
        pass

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group["lr"] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group["lr"] for group in optim.param_groups]
        else:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return lr

    def current_momentum(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """

        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if "momentum" in group.keys():
                    momentums.append(group["momentum"])
                elif "betas" in group.keys():
                    momentums.append(group["betas"][0])
                else:
                    momentums.append(0)
            return momentums

        if self.optimizer is None:
            raise RuntimeError(
                "momentum is not applicable because optimizer does not exist."
            )
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(self.optimizer)
        elif isinstance(self.optimizer, dict):
            momentums = dict()
            for name, optim in self.optimizer.items():
                momentums[name] = _get_momentum(optim)
        return momentums

    def register_hook(
        self, hook: Hook, priority: Union[int, str, Priority] = "NORMAL"
    ) -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name: str) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def get_hook_info(self) -> str:
        # Get hooks info in each stage
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name  # type: ignore
            except ValueError:
                priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f"({priority:<12}) {classname:<35}"
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f"{stage}:\n"
                info += "\n".join(hook_infos)
                info += "\n -----------------------------------------"
                stage_hook_infos.append(info)
        return "\n".join(stage_hook_infos)

    def load_checkpoint(
        self,
        filename: str,
        map_location: Union[str, Callable] = "cpu",
        strict: bool = False,
        revise_keys: List = [(r"^module.", "")],
    ) -> Union[Dict, OrderedDict]:
        return load_checkpoint(
            self.model,
            filename,
            map_location,
            strict,
            self.logger,
            revise_keys=revise_keys,
        )

    @no_type_check
    def resume(
        self,
        checkpoint: str,
        resume_optimizer: bool = True,
        map_location: Union[str, Callable] = "default",
    ) -> None:
        if map_location == "default":
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id),
                )
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)

        self._epoch = checkpoint["meta"]["epoch"]
        self._iter = checkpoint["meta"]["iter"]
        if self.meta is None:
            self.meta = {}
        self.meta.setdefault("hook_msgs", {})
        # load `last_ckpt`, `best_score`, `best_ckpt`, etc. for hook messages
        self.meta["hook_msgs"].update(checkpoint["meta"].get("hook_msgs", {}))

        # resume meta information meta
        self.meta = checkpoint["meta"]

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

        self.logger.info("resumed epoch %d, iter %d", self.epoch, self.iter)

    def register_lr_hook(self, lr_config: Union[Dict, Hook, None]) -> None:
        if lr_config is None:
            return
        elif isinstance(lr_config, dict):
            assert "policy" in lr_config
            policy_type = lr_config.pop("policy")
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater.
            # Since this is not applicable for `
            # CosineAnnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + "LrUpdaterHook"
            lr_config["type"] = hook_type
            hook = build_module(lr_config)
        else:
            hook = lr_config
        self.register_hook(hook, priority="VERY_HIGH")

    def register_optimizer_hook(
        self, optimizer_config: Union[Dict, Hook, None]
    ) -> None:
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault("type", "OptimizerHook")
            hook = build_module(optimizer_config)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority="ABOVE_NORMAL")

    def register_checkpoint_hook(
        self, checkpoint_config: Union[Dict, Hook, None]
    ) -> None:
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault("type", "CheckpointHook")
            hook = build_module(checkpoint_config)
        else:
            hook = checkpoint_config
        self.register_hook(hook, priority="NORMAL")

    def register_logger_hooks(self, log_config: Optional[Dict]) -> None:
        if log_config is None:
            return
        log_interval = log_config["interval"]
        for info in log_config["hooks"]:
            logger_hook = build_module(info, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority="VERY_LOW")

    def register_timer_hook(
        self,
        timer_config: Union[Dict, Hook, None],
    ) -> None:
        if timer_config is None:
            return
        if isinstance(timer_config, dict):
            timer_config_ = timer_config.copy()
            hook = build_module(timer_config_)
        else:
            hook = timer_config
        self.register_hook(hook, priority="LOW")

    def register_training_hooks(
        self,
        lr_config: Union[Dict, Hook, None],
        optimizer_config: Union[Dict, Hook, None] = None,
        checkpoint_config: Union[Dict, Hook, None] = None,
        log_config: Optional[Dict] = None,
        timer_config: Union[Dict, Hook] = dict(type="IterTimerHook"),
    ) -> None:
        """Register default and custom hooks for training.

        Default and custom hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        self.register_lr_hook(lr_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_timer_hook(timer_config)
        self.register_logger_hooks(log_config)


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)

# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
from math import cos, pi
from typing import List, Optional, Union

from .hook import Hook
from tool import runner

__all__ = ["LrUpdaterHook", "CosineAnnealingLrUpdaterHook"]


class LrUpdaterHook(Hook):
    """LR Scheduler in E2E.

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(
        self,
        by_epoch: bool = True,
        warmup: Optional[str] = None,
        warmup_iters: int = 0,
        warmup_ratio: float = 0.1,
        warmup_by_epoch: bool = False,
    ) -> None:
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ["constant", "linear", "exp"]:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant", "linear" and "exp"'
                )
        if warmup is not None:
            assert warmup_iters > 0, '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters: Optional[int] = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs: Optional[int] = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr: Union[list, dict] = []  # initial lr for all param groups
        self.regular_lr: list = []  # expected lr if no warming up is performed

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float):
        raise NotImplementedError

    def get_regular_lr(self, runner: "runner.BaseRunner"):
        if isinstance(runner.optimizer, dict):
            lr_groups = {}
            for k in runner.optimizer.keys():
                _lr_group = [
                    self.get_lr(runner, _base_lr) for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters: int):
        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == "constant":
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == "linear":
                k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == "exp":
                k = self.warmup_ratio ** (1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, runner: "runner.BaseRunner"):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    group.setdefault("initial_lr", group["lr"])
                _base_lr = [group["initial_lr"] for group in optim.param_groups]
                self.base_lr.update({k: _base_lr})
        else:
            for group in runner.optimizer.param_groups:  # type: ignore
                group.setdefault("initial_lr", group["lr"])
            self.base_lr = [
                group["initial_lr"]
                for group in runner.optimizer.param_groups  # type: ignore
            ]

    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group["lr"] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def before_train_epoch(self, runner: "runner.BaseRunner"):
        if self.warmup_iters is None:
            epoch_len = len(runner.data_loader)  # type: ignore
            self.warmup_iters = self.warmup_epochs * epoch_len  # type: ignore

        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner: "runner.BaseRunner"):
        cur_iter = runner.iter
        assert isinstance(self.warmup_iters, int)
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


class CosineAnnealingLrUpdaterHook(LrUpdaterHook):
    """CosineAnnealing LR scheduler.

    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(
        self,
        min_lr: Optional[float] = None,
        min_lr_ratio: Optional[float] = None,
        **kwargs,
    ) -> None:
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    @staticmethod
    def annealing_cos(
        start: float, end: float, factor: float, weight: float = 1.0
    ) -> float:
        """Calculate annealing cos learning rate.

        Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
        percentage goes from 0.0 to 1.0.

        Args:
            start (float): The starting learning rate of the cosine annealing.
            end (float): The ending learing rate of the cosine annealing.
            factor (float): The coefficient of `pi` when calculating the current
                percentage. Range from 0.0 to 1.0.
            weight (float, optional): The combination factor of `start` and `end`
                when calculating the actual starting learning rate. Default to 1.
        """
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr  # type:ignore
        return CosineAnnealingLrUpdaterHook.annealing_cos(
            base_lr, target_lr, progress / max_progress
        )


class StepLrUpdaterHook(LrUpdaterHook):
    """Step LR scheduler with min_lr clipping.

    Args:
        step (int | list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (float): Decay LR ratio. Defaults to 0.1.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
    """

    def __init__(
        self,
        step: Union[int, List[int]],
        gamma: float = 0.1,
        min_lr: Optional[float] = None,
        **kwargs,
    ) -> None:
        if isinstance(step, list):
            assert all([s > 0 for s in step])
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float):
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr

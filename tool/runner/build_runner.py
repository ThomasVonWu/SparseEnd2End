# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
from typing import Optional, Dict, Any
from .iter_based_runner import *


__all__ = ["build_runner"]


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def build_runner_constructor(cfg: dict):
    return build_module(cfg)


def build_runner(cfg: dict, default_args: Optional[dict] = None):
    runner_cfg = cfg.copy()
    constructor_type = runner_cfg.pop("constructor", "DefaultRunnerConstructor")
    runner_constructor = build_runner_constructor(
        dict(type=constructor_type, runner_cfg=runner_cfg, default_args=default_args)
    )
    runner = runner_constructor()
    return runner


class DefaultRunnerConstructor:
    """Default constructor for runners.
    Custom existing `Runner` like `EpocBasedRunner` though `RunnerConstructor`.
    For example, We can inject some new properties and functions for `Runner`.
    """

    def __init__(self, runner_cfg: dict, default_args: Optional[dict] = None):
        if not isinstance(runner_cfg, dict):
            raise TypeError(
                "runner_cfg should be a dict", f"but got {type(runner_cfg)}"
            )
        self.runner_cfg = runner_cfg
        self.default_args = default_args

    def __call__(self):
        return build_module(self.runner_cfg, default_args=self.default_args)

# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import warnings
import pkgutil
import torch.nn as nn

from torch.optim import *
from torch.nn import GroupNorm, LayerNorm
from typing import Dict, List, Optional, Union, Any
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

__all__ = ["build_optimizer", "DefaultOptimizerConstructor"]


class DefaultOptimizerConstructor:
    def __init__(self, optimizer_cfg: Dict, paramwise_cfg: Optional[Dict] = None):
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg
        self.base_lr = optimizer_cfg.get("lr", None)
        self.base_wd = optimizer_cfg.get("weight_decay", None)

    def _is_in(self, param_group: Dict, param_group_list: List) -> bool:
        param = set(param_group["params"])
        param_set = set()
        for group in param_group_list:
            param_set.update(set(group["params"]))

        return not param.isdisjoint(param_set)

    def add_params(
        self,
        params: List[Dict],
        module: nn.Module,
        prefix: str = "",
        is_dcn_module: Union[int, float, None] = None,
    ) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get("custom_keys", {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get("bias_lr_mult", 1.0)
        bias_decay_mult = self.paramwise_cfg.get("bias_decay_mult", 1.0)
        norm_decay_mult = self.paramwise_cfg.get("norm_decay_mult", 1.0)
        dwconv_decay_mult = self.paramwise_cfg.get("dwconv_decay_mult", 1.0)
        bypass_duplicate = self.paramwise_cfg.get("bypass_duplicate", False)
        dcn_offset_lr_mult = self.paramwise_cfg.get("dcn_offset_lr_mult", 1.0)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d) and module.in_channels == module.groups
        )

        for name, param in module.named_parameters(recurse=False):
            param_group = {"params": [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                warnings.warn(
                    f"{prefix} is duplicate. It is skipped since "
                    f"bypass_duplicate={bypass_duplicate}"
                )
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f"{prefix}.{name}":
                    is_custom = True
                    lr_mult = custom_keys[key].get("lr_mult", 1.0)
                    param_group["lr"] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get("decay_mult", 1.0)
                        param_group["weight_decay"] = self.base_wd * decay_mult
                    break

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if name == "bias" and not (is_norm or is_dcn_module):
                    param_group["lr"] = self.base_lr * bias_lr_mult

                if (
                    prefix.find("conv_offset") != -1
                    and is_dcn_module
                    and isinstance(module, torch.nn.Conv2d)
                ):
                    # deal with both dcn_offset's bias & weight
                    param_group["lr"] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm:
                        param_group["weight_decay"] = self.base_wd * norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group["weight_decay"] = self.base_wd * dwconv_decay_mult
                    # bias lr and decay
                    elif name == "bias" and not is_dcn_module:
                        # TODO: current bias_decay_mult will have affect on DCN
                        param_group["weight_decay"] = self.base_wd * bias_decay_mult
            params.append(param_group)

        if pkgutil.find_loader("e2edeform_conv2d_ext") is not None:
            from modules.ops import DeformConv2d

            is_dcn_module = isinstance(module, (DeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self.add_params(
                params, child_mod, prefix=child_prefix, is_dcn_module=is_dcn_module
            )

    def __call__(self, model: nn.Module):
        if hasattr(model, "module"):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg["params"] = model.parameters()
            return build_module(optimizer_cfg)

        # set param-wise lr and weight decay recursively
        params: List[Dict] = []
        self.add_params(params, model)
        optimizer_cfg["params"] = params

        return build_module(optimizer_cfg)


def build_optimizer(model, cfg: Dict):

    optimizer_cfg = cfg.copy()
    constructor_type = optimizer_cfg.pop("constructor", "DefaultOptimizerConstructor")
    paramwise_cfg = optimizer_cfg.pop("paramwise_cfg", None)
    optim_constructor = build_module(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg,
        )
    )
    optimizer = optim_constructor(model)
    return optimizer


def build_module(cfg) -> Any:
    cfg2 = cfg.copy()
    type = cfg2.pop("type")
    return eval(type)(**cfg2)

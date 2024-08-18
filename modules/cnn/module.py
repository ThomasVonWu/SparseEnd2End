# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import torch.nn as nn
import numpy as np

from typing import Union, Dict, List, Optional, Any
from tool.utils.logging import get_logger, print_log
from tool.runner.checkpoint import load_checkpoint


def xavier_init(
    module: nn.Module, gain: float = 1, bias: float = 0, distribution: str = "normal"
) -> None:
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
    distribution: str = "normal",
) -> None:
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(
    module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob: float) -> float:
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def initialize(module: nn.Module, init_cfg: Union[Dict, List[dict]]) -> None:
    r"""Initialize a module.

    Args:
        module (``torch.nn.Module``): the module will be initialized.
        init_cfg (dict | list[dict]): initialization configuration dict to
            define initializer. E2E has implemented 6 initializers
            including ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
            ``Kaiming``, and ``Pretrained``.

    Example:
        >>> module = nn.Linear(2, 3, bias=True)
        >>> init_cfg = dict(type='Constant', layer='Linear', val =1 , bias =2)
        >>> initialize(module, init_cfg)

        >>> module = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))
        >>> # define key ``'layer'`` for initializing layer with different
        >>> # configuration
        >>> init_cfg = [dict(type='Constant', layer='Conv1d', val=1),
                dict(type='Constant', layer='Linear', val=2)]
        >>> initialize(module, init_cfg)

        >>> # define key``'override'`` to initialize some specific part in
        >>> # module
        >>> class FooNet(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.feat = nn.Conv2d(3, 16, 3)
        >>>         self.reg = nn.Conv2d(16, 10, 3)
        >>>         self.cls = nn.Conv2d(16, 5, 3)
        >>> model = FooNet()
        >>> init_cfg = dict(type='Constant', val=1, bias=2, layer='Conv2d',
        >>>     override=dict(type='Constant', name='reg', val=3, bias=4))
        >>> initialize(model, init_cfg)

        >>> model = ResNet(depth=50)
        >>> # Initialize weights with the pretrained model.
        >>> init_cfg = dict(type='Pretrained',
                checkpoint='torchvision://resnet50')
        >>> initialize(model, init_cfg)
    """

    def _initialize(module: nn.Module, cfg: Dict, wholemodule: bool = False) -> None:
        func = build_module(cfg)
        # wholemodule flag is for override mode, there is no layer key in override
        # and initializer will give init values for the whole module with the name
        # in override.
        func.wholemodule = wholemodule
        func(module)

    if not isinstance(init_cfg, (dict, list)):
        raise TypeError(
            f"init_cfg must be a dict or a list of dict, \
                but got {type(init_cfg)}"
        )

    if isinstance(init_cfg, dict):
        init_cfg = [init_cfg]

    for cfg in init_cfg:
        # should deeply copy the original config because cfg may be used by
        # other modules, e.g., one init_cfg shared by multiple bottleneck
        # blocks, the expected cfg will be changed after pop and will change
        # the initialization behavior of other modules
        cp_cfg = cfg.copy()
        _initialize(module, cp_cfg)


# PretrainedInit
class Pretrained:
    """Initialize module by loading a pretrained model.

    Args:
        checkpoint (str): the checkpoint file of the pretrained model should
            be load.
        prefix (str, optional): the prefix of a sub-module in the pretrained
            model. it is for loading a part of the pretrained model to
            initialize. For example, if we would like to only load the
            backbone of a detector model, we can set ``prefix='backbone.'``.
            Defaults to None.
        map_location (str): map tensors into proper locations.
    """

    def __init__(self, checkpoint: str, map_location: Optional[str] = None):
        self.checkpoint = checkpoint
        self.map_location = map_location

    def __call__(self, module: nn.Module) -> None:
        logger = get_logger("E2E")
        print_log(f"load model from: {self.checkpoint}", logger=logger)
        load_checkpoint(
            module,
            self.checkpoint,
            map_location=self.map_location,
            strict=False,
            logger=logger,
        )

        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        info = f"{self.__class__.__name__}: load from {self.checkpoint}"
        return info


class BaseInit:
    def __init__(
        self,
        *,
        bias: float = 0,
        bias_prob: Optional[float] = None,
        layer: Union[str, List, None] = None,
    ):
        self.wholemodule = False
        if not isinstance(bias, (int, float)):
            raise TypeError(f"bias must be a number, but got a {type(bias)}")

        if bias_prob is not None:
            if not isinstance(bias_prob, float):
                raise TypeError(
                    f"bias_prob type must be float, \
                    but got {type(bias_prob)}"
                )

        if layer is not None:
            if not isinstance(layer, (str, list)):
                raise TypeError(
                    f"layer must be a str or a list of str, \
                    but got a {type(layer)}"
                )
        else:
            layer = []

        if bias_prob is not None:
            self.bias = bias_init_with_prob(bias_prob)
        else:
            self.bias = bias
        self.layer = [layer] if isinstance(layer, str) else layer

    def _get_init_info(self) -> str:
        info = f"{self.__class__.__name__}, bias={self.bias}"
        return info


class Xavier(BaseInit):
    r"""Initialize module parameters with values according to the method
    described in `Understanding the difficulty of training deep feedforward.

    neural networks - Glorot, X. & Bengio, Y. (2010).
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_

    Args:
        gain (int | float): an optional scaling factor. Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'``
            or ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, gain: float = 1, distribution: str = "normal", **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        self.distribution = distribution

    def __call__(self, module: nn.Module) -> None:
        def init(m):
            if self.wholemodule:
                xavier_init(m, self.gain, self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    xavier_init(m, self.gain, self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        info = (
            f"{self.__class__.__name__}: gain={self.gain}, "
            f"distribution={self.distribution}, bias={self.bias}"
        )
        return info


def _get_bases_name(m: nn.Module) -> List[str]:
    return [b.__name__ for b in m.__class__.__bases__]


def update_init_info(module: nn.Module, init_info: str) -> None:
    """Update the `_params_init_info` in the module if the value of parameters
    are changed.

    Args:
        module (obj:`nn.Module`): The module of PyTorch with a user-defined
            attribute `_params_init_info` which records the initialization
            information.
        init_info (str): The string that describes the initialization.
    """
    assert hasattr(
        module, "_params_init_info"
    ), f"Can not find `_params_init_info` in {module}"
    for name, param in module.named_parameters():

        assert param in module._params_init_info, (
            f"Find a new :obj:`Parameter` "
            f"named `{name}` during executing the "
            f"`init_weights` of "
            f"`{module.__class__.__name__}`. "
            f"Please do not add or "
            f"replace parameters during executing "
            f"the `init_weights`. "
        )

        # The parameter has been changed during executing the
        # `init_weights` of module
        mean_value = param.data.mean()
        if module._params_init_info[param]["tmp_mean_value"] != mean_value:
            module._params_init_info[param]["init_info"] = init_info
            module._params_init_info[param]["tmp_mean_value"] = mean_value


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

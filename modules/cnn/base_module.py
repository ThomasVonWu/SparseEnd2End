# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import copy
import torch
import functools
import warnings
import torch.nn as nn


from abc import ABCMeta
from typing import Optional, Callable, Union, Tuple, Dict
from collections import defaultdict
from tool.utils.dist_utils import get_dist_info
from tool.utils.logging import get_logger, print_log
from logging import FileHandler
from modules.cnn.module import kaiming_init, initialize

logger_initialized: dict = {}


def master_only(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


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


class BaseModule(nn.Module, metaclass=ABCMeta):
    """
    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

    - ``init_cfg``: the config to control the initialization.
    - ``init_weights``: The function of parameter initialization and recording
      initialization information.
    - ``_params_init_info``: Used to track the parameter initialization
      information. This attribute only exists during executing the
      ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, init_cfg: Optional[dict] = None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        super().__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self) -> bool:
        return self._is_init

    def init_weights(self) -> None:
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, "_params_init_info"):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info: defaultdict = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param]["init_info"] = (
                    f"The value is the same before and "
                    f"after calling `init_weights` "
                    f"of {self.__class__.__name__} "
                )
                self._params_init_info[param]["tmp_mean_value"] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `E2E`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else "E2E"

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f"initialize {module_name} with init_cfg {self.init_cfg}",
                    logger=logger_name,
                )
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg["type"] == "Pretrained":
                        return

            for m in self.children():
                if hasattr(m, "init_weights"):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f"Initialized by "
                        f"user-defined `init_weights`"
                        f" in {m.__class__.__name__} ",
                    )

            self._is_init = True
        else:
            warnings.warn(
                f"init_weights of {self.__class__.__name__} has "
                f"been called more than once."
            )

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    @master_only
    def _dump_init_info(self, logger_name: str) -> None:
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.

        Args:
            logger_name (str): The name of logger.
        """

        logger = get_logger(logger_name)

        with_file_handler = True
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                # handler.stream.write("Name of parameter - Initialization information\n")
                # for name, param in self.named_parameters():
                #     handler.stream.write(
                #         f"\n{name} - {param.shape}: "
                #         f"\n{self._params_init_info[param]['init_info']} \n"
                #     )
                # handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                print_log(
                    f"\n{name} - {param.shape}: "
                    f"\n{self._params_init_info[param]['init_info']} \n ",
                    logger=logger_name,
                )

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


class Sequential(BaseModule, nn.Sequential):
    def __init__(self, *args, init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Union[bool, str] = "auto",
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = dict(type="ReLU"),
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = "zeros",
        order: tuple = ("conv", "norm", "act"),
    ):
        super().__init__()
        assert conv_cfg is None
        assert norm_cfg is None
        assert act_cfg is None
        official_padding_mode = ["zeros", "circular"]
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        # Current ConvModule only support zero and curcular
        self.with_explicit_padding = padding_mode not in official_padding_mode
        assert not self.with_explicit_padding, "[Error] Only Support Zero And Curcular!"
        self.order = order

        assert isinstance(self.order, tuple) and len(self.order) <= 3

        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.

        assert bias == "auto", "[Error] Current ConvModule Support Only Bias Auto!"
        self.with_bias = bias

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding

        ## Step1 build convolution layer
        self.conv = nn.modules.conv.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Step2 build normalization layers
        pass

        # Step3 build activation layer
        pass

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, "init_weights"):
            nonlinearity = "relu"
            a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
        return x

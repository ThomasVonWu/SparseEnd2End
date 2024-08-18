# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch.nn as nn


def wrap_fp16_model(model: nn.Module) -> None:
    """Wrap the FP32 model to FP16.

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend.

    For PyTorch >= 1.6, this function will
    1. Set fp16 flag inside the model to True.

    Otherwise:
    1. Convert FP32 model to FP16.
    2. Remain some necessary layers to be FP32, e.g., normalization layers.
    3. Set `fp16_enabled` flag inside the model to True.

    Args:
        model (nn.Module): Model in FP32.
    """
    # set `fp16_enabled` flag
    for m in model.modules():
        if hasattr(m, "fp16_enabled"):
            m.fp16_enabled = True

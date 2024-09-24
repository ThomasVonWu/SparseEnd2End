# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

import os
import numpy as np
from typing import List


def save_bin_format(x, x_name: str, save_prefix, sample_index, logger):
    """x : numpy.ndarray"""
    logger.debug(
        f"{x_name}\t:\t{x.flatten()[:5]} ... {x.flatten()[-5:]}, min={x.min()}, max={x.max()}"
    )

    x_shape = "*".join([str(it) for it in x.shape])
    x_path = os.path.join(
        save_prefix,
        f"sample_{sample_index}_{x_name}_{x_shape}_{x.dtype}.bin",
    )
    x.tofile(x_path)
    logger.info(f"Save data bin file: {x_path}.")


def save_bins(
    inputs: List[np.ndarray],
    outputs: List[np.ndarray],
    names: str,
    sample_index: int,
    logger,
    save_prefix: str = "script/tutorial/asset",
):
    os.makedirs(save_prefix, exist_ok=True)
    assert len(inputs + outputs) == len(names)

    for x, x_name in zip(inputs + outputs, names):
        save_bin_format(x, x_name, save_prefix, sample_index, logger)

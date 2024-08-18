# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import torch
import functools
import torch.nn as nn

from typing import Tuple, Callable, List
from collections import OrderedDict
from torch import distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch._utils import _flatten_dense_tensors, _take_tensors, _unflatten_dense_tensors


def init_dist(launcher: str, backend: str = "nccl", **kwargs) -> None:
    def _init_dist_pytorch(backend: str, **kwargs) -> None:
        # TODO: use local_rank instead of rank % num_gpus
        rank = int(os.environ["RANK"])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        # 进程间打开通信
        dist.init_process_group(backend=backend, **kwargs)

    if torch.multiprocessing.get_start_method(allow_none=True) is None:
        # 使用spawn启动多进程
        torch.multiprocessing.set_start_method("spawn")
    if launcher == "pytorch":
        _init_dist_pytorch(backend, **kwargs)
    else:
        raise ValueError(f"Invalid launcher type: {launcher}")


def reduce_mean(tensor):
    """ "Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def is_module_wrapper(module: nn.Module) -> bool:
    """Check if a module is a module wrapper.

    The following 2 modules in  E2E (and their subclasses) are regarded as
    module wrappers: DataParallel, DistributedDataParallel.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the input module is a module wrapper.
    """

    def is_module_in_wrapper(module):
        module_wrappers = (DataParallel, DistributedDataParallel)
        if isinstance(module, module_wrappers):
            return True
        return False

    return is_module_in_wrapper(module)


def allreduce_params(
    params: List[torch.nn.Parameter], coalesce: bool = True, bucket_size_mb: int = -1
) -> None:
    """Allreduce parameters.

    Args:
        params (list[torch.nn.Parameter]): List of parameters or buffers
            of a model.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """

    def _allreduce_coalesced(
        tensors: torch.Tensor, world_size: int, bucket_size_mb: int = -1
    ) -> None:
        if bucket_size_mb > 0:
            bucket_size_bytes = bucket_size_mb * 1024 * 1024
            buckets = _take_tensors(tensors, bucket_size_bytes)
        else:
            buckets = OrderedDict()
            for tensor in tensors:
                tp = tensor.type()
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(tensor)
            buckets = buckets.values()

        for bucket in buckets:
            flat_tensors = _flatten_dense_tensors(bucket)
            dist.all_reduce(flat_tensors)
            flat_tensors.div_(world_size)
            for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)
            ):
                tensor.copy_(synced)

    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params, world_size, bucket_size_mb)
    else:
        for tensor in params:
            dist.all_reduce(tensor.div_(world_size))

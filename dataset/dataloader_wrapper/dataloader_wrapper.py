# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import random
import numpy as np

from functools import partial
from typing import Optional, Any, Dict

from torch.utils.data import DataLoader
from tool.utils.dist_utils import get_dist_info

from dataset.sampler import *
from ..utils.collate import collate_fn

__all__ = ["dataloader_wrapper", "dataloader_wrapper_without_dist"]


def _worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


def dataloader_wrapper(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    seed=None,
    runner_type="EpochBasedRunner",
    dist=True,
    num_gpus=1,
    shuffle=True,
    nonshuffler_sampler=None,
    **kwargs
):
    """Wrapper PyTorch DataLoader.

    Temporarily not support to DistributeTest.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """

    # def _worker_init_fn(worker_id, num_workers, rank, seed):
    #     worker_seed = num_workers * rank + worker_id + seed
    #     random.seed(worker_seed)
    #     np.random.seed(worker_seed)
    #     torch.manual_seed(worker_seed)
    #     torch.cuda.manual_seed(worker_seed)
    #     torch.cuda.manual_seed_all(worker_seed)

    rank, world_size = get_dist_info()
    batch_sampler = None
    if runner_type == "IterBasedRunner":  # dist train
        batch_sampler = GroupInBatchSampler(
            dataset,
            samples_per_gpu,
            world_size,
            rank,
            seed=seed,
        )
        sampler = None
        batch_size = 1
        shuffle = None
        drop_last = False
        num_workers = workers_per_gpu
    elif dist:  # dist val
        assert not shuffle
        sampler = build_module(
            (
                nonshuffler_sampler
                if nonshuffler_sampler is not None
                else dict(type="DistributedSampler")
            ),
            dict(
                dataset=dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
            ),
        )
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        assert not shuffle
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        shuffle = False
        num_workers = num_gpus * workers_per_gpu

    init_fn = (
        partial(_worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate_fn, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs
    )

    return data_loader


def dataloader_wrapper_without_dist(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    num_gpus=1,
    seed=None,
    dist=False,
    shuffle=False,
    runner_type="EpochBasedRunner",
    persistent_workers=False,
    **kwargs
):
    """Build PyTorch DataLoader.
    In non-distributed test, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: False.
        seed (int, Optional): Seed to be used. Default: None.
        runner_type (str): Type of runner. Default: `EpochBasedRunner`
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers `Dataset` instances alive.
            This argument is only valid when PyTorch>=1.7.0. Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """

    # def _worker_init_fn(worker_id, num_workers, rank, seed):
    #     # The seed of each worker equals to
    #     # num_worker * rank + worker_id + user_seed
    #     worker_seed = num_workers * rank + worker_id + seed
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)
    #     torch.manual_seed(worker_seed)

    rank, world_size = get_dist_info()

    # When model is obj:`DataParallel`
    # the batch size is samples on all the GPUS
    batch_size = num_gpus * samples_per_gpu
    num_workers = num_gpus * workers_per_gpu

    sampler = None
    batch_sampler = None

    init_fn = (
        partial(_worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    kwargs["persistent_workers"] = persistent_workers

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate_fn, samples_per_gpu=samples_per_gpu),
        pin_memory=kwargs.pop("pin_memory", False),
        worker_init_fn=init_fn,
        **kwargs
    )

    return data_loader


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)

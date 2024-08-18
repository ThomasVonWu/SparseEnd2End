# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
"""This file holding some environment constant for sharing by other files."""
import cv2
import sys
import torch
import subprocess
import os.path as osp

from collections import defaultdict


def collect_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - MSVC: Microsoft Virtual C++ Compiler version, Windows only.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
    """
    env_info = {}
    env_info["sys.platform"] = sys.platform
    env_info["Python"] = sys.version.replace("\n", "")

    cuda_available = torch.cuda.is_available()
    env_info["CUDA available"] = cuda_available

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info["GPU " + ",".join(device_ids)] = name

        from torch.utils.cpp_extension import CUDA_HOME

        env_info["CUDA_HOME"] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, "bin/nvcc")
                nvcc = subprocess.check_output(f'"{nvcc}" -V', shell=True)
                nvcc = nvcc.decode("utf-8").strip()
                release = nvcc.rfind("Cuda compilation tools")
                build = nvcc.rfind("Build ")
                nvcc = nvcc[release:build].strip()
            except subprocess.SubprocessError:
                nvcc = "Not Available"
            env_info["NVCC"] = nvcc

    try:
        # Check C++ Compiler.
        # For Unix-like, sysconfig has 'CC' variable like 'gcc -pthread ...',
        # indicating the compiler used, we use this to get the compiler name
        import sysconfig

        cc = sysconfig.get_config_var("CC")
        if cc:
            cc = osp.basename(cc.split()[0])
            cc_info = subprocess.check_output(f"{cc} --version", shell=True)
            env_info["GCC"] = cc_info.decode("utf-8").partition("\n")[0].strip()
        else:
            # on Windows, cl.exe is not in PATH. We need to find the path.
            # distutils.ccompiler.new_compiler() returns a msvccompiler
            # object and after initialization, path to cl.exe is found.
            import locale
            import os
            from distutils.ccompiler import new_compiler

            ccompiler = new_compiler()
            ccompiler.initialize()
            cc = subprocess.check_output(
                f"{ccompiler.cc}", stderr=subprocess.STDOUT, shell=True
            )
            encoding = (
                os.device_encoding(sys.stdout.fileno()) or locale.getpreferredencoding()
            )
            env_info["MSVC"] = cc.decode(encoding).partition("\n")[0].strip()
            env_info["GCC"] = "n/a"
    except subprocess.CalledProcessError:
        env_info["GCC"] = "n/a"

    env_info["PyTorch"] = torch.__version__
    env_info["PyTorch compiling details"] = torch.__config__.show()

    try:
        import torchvision

        env_info["TorchVision"] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    env_info["OpenCV"] = cv2.__version__

    return env_info

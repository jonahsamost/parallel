import importlib
from parallel.engine.utils.environment import patch_environment
import torch


def _is_package_available(pkg_name):
    pkg = importlib.util.find_spec(pkg_name)
    if pkg is not None:
        try:
            _ = importlib.metadata.metadata(pkg_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

def is_wandb_available():
    return _is_package_available("wandb")


def is_deepspeed_available():
    return _is_package_available("deepspeed")

def is_torchao_available():
    return _is_package_available("torchao")

def is_fp8_available():
    return is_torchao_available()

def is_datasets_available():
    return _is_package_available("datasets")

def is_torchdata_available():
    return _is_package_available("torchdata")


def is_cuda_available():
    """
    Checks if `cuda` is available via an `nvml-based` check which won't trigger the drivers and leave cuda
    uninitialized.
    """
    with patch_environment(PYTORCH_NVML_BASED_CUDA_CHECK="1"):
        available = torch.cuda.is_available()

    return available
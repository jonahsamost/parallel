import random
import numpy as np
import torch
import os
from pathlib import Path
import sys
from omegaconf import DictConfig, OmegaConf
from contextlib import contextmanager
import torch.distributed as dist

from parallel.state import ParallelConfig, RuntimeState, Strategies

DTYPE_DICT = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

def empty_fn(*args, **kwargs):
    pass

def load_cfg(path: Path | None = None) -> DictConfig:
    if path is None:
        path = Path(__file__).parent / "conf" / "default.yaml"
    base = OmegaConf.load(path)
    assert isinstance(base, DictConfig)

    args = sys.argv[1:]
    yaml_args = [a for a in args if a.endswith((".yaml", ".yml"))]
    dot_overrides = [a for a in args if "=" in a and not a.endswith((".yaml", ".yml"))]

    if yaml_args:
        override = OmegaConf.load(yaml_args[0])
        cfg = OmegaConf.merge(base, override)
    else:
        cfg = base

    if dot_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dot_overrides))

    assert isinstance(cfg, DictConfig)
    return cfg


@contextmanager
def patch_environment(**kwargs):
    existing_vars = {}
    for key, value in kwargs.items():
        key = key.upper()
        if key in os.environ:
            existing_vars[key] = os.environ[key]
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key in kwargs:
            key = key.upper()
            if key in existing_vars:
                # restore previous value
                os.environ[key] = existing_vars[key]
            else:
                os.environ.pop(key, None)

def is_cuda_available():
    with patch_environment(PYTORCH_NVML_BASED_CUDA_CHECK="1"):
        available = torch.cuda.is_available()
    return available


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def dist_cleanup():
    if is_dist_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _mesh_rank(pconfig: ParallelConfig, dim_name: str) -> int:
    if pconfig.device_mesh is not None and dim_name in pconfig.device_mesh.mesh_dim_names:
        return pconfig.device_mesh.get_local_rank(dim_name)
    return 0


def model_init_rngs(pconfig: ParallelConfig, seed: int = 42):
    """
    different rank TP get different seeds
    """
    tp_rank = _mesh_rank(pconfig, Strategies.TP)
    init_seed = seed + tp_rank
    torch.manual_seed(init_seed)
    if pconfig.device_type == "cuda":
        torch.cuda.manual_seed_all(init_seed)


def model_train_rngs(pconfig: ParallelConfig, seed: int = 42):
    """
    - DP ranks get different seeds (each sees different data)
    - TP ranks within same (DP/PP) group get same seed
    """
    dp_rank = _mesh_rank(pconfig, Strategies.DP_REPLICATE)
    pp_rank = _mesh_rank(pconfig, Strategies.PP)

    data_seed = seed + dp_rank
    random.seed(data_seed)
    np.random.seed(data_seed)

    torch_seed = seed + dp_rank * 1000 + pp_rank
    torch.manual_seed(torch_seed)
    if pconfig.device_type == "cuda":
        torch.cuda.manual_seed_all(torch_seed)
import torch
import os
from pathlib import Path
import sys
from omegaconf import DictConfig, OmegaConf
from contextlib import contextmanager



def load_cfg(path: Path) -> DictConfig:
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

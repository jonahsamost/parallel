import random
from typing import Optional, Union

import numpy as np
import torch

from parallel.engine.state import RuntimeState
from parallel.engine.utils.dataclasses import DistType, RNGType

_CUDA_DIST_TYPES = {DistType.MULTI_GPU, DistType.DEEPSPEED, DistType.FSDP, DistType.FLAM}


def set_seed(seed: int, device_specific: bool = False, deterministic: bool = False):
    """
    Set the seed in `random`, `numpy`, and `torch` for reproducible runs.
    """
    if device_specific:
        seed += RuntimeState().process_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def synchronize_rng_state(rng_type: Optional[RNGType] = None, generator: Optional[torch.Generator] = None):
    state = RuntimeState()
    if state.distributed_type == DistType.NONE:
        return

    if rng_type == RNGType.TORCH:
        rng_state = torch.get_rng_state()
    elif rng_type == RNGType.CUDA:
        rng_state = torch.cuda.get_rng_state()
    elif rng_type == RNGType.GENERATOR:
        if generator is None:
            raise AssertionError("Need a generator to synchronize its seed.")
        rng_state = generator.get_state()
    else:
        raise ValueError(f"Unsupported RNG type: {rng_type}")

    if state.distributed_type == DistType.MULTI_CPU:
        torch.distributed.broadcast(rng_state, 0)
    elif state.distributed_type in _CUDA_DIST_TYPES:
        rng_state = rng_state.to(state.device)
        torch.distributed.broadcast(rng_state, 0)
        rng_state = rng_state.cpu()
    else:
        return

    if rng_type == RNGType.TORCH:
        torch.set_rng_state(rng_state)
    elif rng_type == RNGType.CUDA:
        torch.cuda.set_rng_state(rng_state)
    elif rng_type == RNGType.GENERATOR:
        generator.set_state(rng_state)


def synchronize_rng_states(rng_types: list[Union[str, RNGType]], generator: Optional[torch.Generator] = None):
    for rng_type in rng_types:
        synchronize_rng_state(RNGType(rng_type), generator=generator)

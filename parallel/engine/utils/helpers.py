import torch
import collections
from collections import OrderedDict
from safetensors.torch import save_file as safe_save_file
from functools import partial, reduce
from parallel.engine.state import (
    RuntimeState, DistType
)
from codecs import encode
from parallel.engine.logging import get_logger
import numpy as np


np_core = np.core
TORCH_SAFE_GLOBALS = [
    # numpy arrays are just numbers, not objects, so we can reconstruct them safely
    np_core.multiarray._reconstruct,
    np.ndarray,
    # The following are needed for the RNG states
    encode,
    np.dtype,
]


logger = get_logger(__name__)


def clean_state_dict_for_safetensors(state_dict: dict):
    """
    Cleans the state dictionary from a model and removes tensor aliasing if present.

    Args:
        state_dict (`dict`):
            The state dictionary from a model
    """
    ptrs = collections.defaultdict(list)
    # When bnb serialization is used, weights in state dict can be strings
    for name, tensor in state_dict.items():
        if not isinstance(tensor, str):
            ptrs[id_tensor_storage(tensor)].append(name)

    # These are all pointers of tensors with shared memory
    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
    warn_names = set()
    for names in shared_ptrs.values():
        # When not all duplicates have been cleaned, we still remove those keys but put a clear warning.
        # If the link between tensors was done at runtime then `from_pretrained` will not get
        # the key back leading to random tensor. A proper warning will be shown
        # during reload (if applicable), but since the file is not necessarily compatible with
        # the config, better show a proper warning.
        found_names = [name for name in names if name in state_dict]
        warn_names.update(found_names[1:])
        for name in found_names[1:]:
            del state_dict[name]
    if len(warn_names) > 0:
        logger.warning(
            f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading",
        )
    state_dict = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    return state_dict


def save(obj, f, save_on_each_node: bool = False, safe_serialization: bool = False):
    """
    Save the data to disk. Use in place of `torch.save()`.

    Args:
        obj:
            The data to save
        f:
            The file (or file-like object) to use to save the data
        save_on_each_node (`bool`, *optional*, defaults to `False`):
            Whether to only save on the global main process
        safe_serialization (`bool`, *optional*, defaults to `False`):
            Whether to save `obj` using `safetensors` or the traditional PyTorch way (that uses `pickle`).
    """
    # Check if it's a model and remove duplicates
    if safe_serialization:
        save_func = partial(safe_save_file, metadata={"format": "pt"})
        if isinstance(obj, OrderedDict):
            obj = clean_state_dict_for_safetensors(obj)
    else:
        save_func = torch.save

    if RuntimeState().is_main_process and not save_on_each_node:
        save_func(obj, f)
    elif RuntimeState().is_local_main_process and save_on_each_node:
        save_func(obj, f)


def load(f, map_location=None, **kwargs):
    """
    Compatible drop-in replacement of `torch.load()` which allows for `weights_only` to be used if `torch` version is
    2.4.0 or higher. Otherwise will ignore the kwarg.

    Will also add (and then remove) an exception for numpy arrays

    Args:
        f:
            The file (or file-like object) to use to load the data
        map_location:
            a function, `torch.device`, string or a dict specifying how to remap storage locations
        **kwargs:
            Additional keyword arguments to pass to `torch.load()`.
    """
    try:
        old_safe_globals = torch.serialization.get_safe_globals()
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = True
        torch.serialization.add_safe_globals(TORCH_SAFE_GLOBALS)
        loaded_obj = torch.load(f, map_location=map_location, **kwargs)
    finally:
        torch.serialization.clear_safe_globals()
        if old_safe_globals:
            torch.serialization.add_safe_globals(old_safe_globals)
    return loaded_obj


def get_pretty_name(obj):
    """
    Gets a pretty name from `obj`.
    """
    if not hasattr(obj, "__qualname__") and not hasattr(obj, "__name__"):
        obj = getattr(obj, "__class__", obj)
    if hasattr(obj, "__qualname__"):
        return obj.__qualname__
    if hasattr(obj, "__name__"):
        return obj.__name__
    return str(obj)
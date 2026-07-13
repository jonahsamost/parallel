import os
from dataclasses import dataclass, field
from contextlib import contextmanager

DEFAULT_MASTER_PORT = 29500

@dataclass
class CPUInfo:
    rank: int = field(default=0, metadata={"help": "The rank of the current process."})
    world_size: int = field(default=1, metadata={"help": "The total number of processes in the world."})
    local_rank: int = field(default=0, metadata={"help": "The rank of the current process on the local node."})
    local_world_size: int = field(default=1, metadata={"help": "The total number of processes on the local node."})


def get_cpu_distributed_information() -> CPUInfo:
    information = {}
    information["rank"] = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0)
    information["world_size"] = get_int_from_env(
        ["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1
    )
    information["local_rank"] = get_int_from_env(
        ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"], 0
    )
    information["local_world_size"] = get_int_from_env(
        ["LOCAL_WORLD_SIZE", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"],
        1,
    )
    return CPUInfo(**information)


def get_int_from_env(env_keys, default):
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default

def str_to_bool(value: str, to_bool: bool = True):
    if not value:
        return False if to_bool else 0
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return True if to_bool else 1
    elif value in ("n", "no", "f", "false", "off", "0"):
        return False if to_bool else 0
    else:
        raise ValueError(f"invalid truth value {value}")

def get_log_level():
    return os.environ.get("ENGINE_LOG_LEVEL", None)

def get_torch_device():
    return os.environ.get("ENGINE_TORCH_DEVICE", None)

def get_debug_mode():
    return str_to_bool(os.environ.get("ENGINE_DEBUG_MODE", None))

def get_use_deepspeed():
    return str_to_bool(os.environ.get("ENGINE_USE_DEEPSPEED", None))

def get_use_fsdp():
    return str_to_bool(os.environ.get("ENGINE_USE_FSDP", None))

def get_fsdp_offload_params():
    return str_to_bool(os.environ.get("ENGINE_FSDP_OFFLOAD_PARAMS", None))

def get_fsdp_state_dict_type():
    return os.environ.get("ENGINE_FSDP_STATE_DICT_TYPE", "SHARDED_STATE_DICT")

def get_master_port():
    return os.environ.get("MASTER_PORT", None)

def get_master_addr():
    return os.environ.get("MASTER_ADDR", None)

def get_omp_num_threads():
    return int(os.environ.get("OMP_NUM_THREADS", 0))

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", -1))

def get_fork_launched():
    return str_to_bool(os.environ.get("FORK_LAUNCHED", None))

def get_use_cpu():
    return str_to_bool(os.environ.get("ENGINE_USE_CPU", None))

def get_mixed_precision():
    return os.environ.get("ENGINE_MIXED_PRECISION", "no")

def get_allow_cp_standalone():
    return str_to_bool(os.environ.get("ENGINE_ALLOW_CP_STANDALONE", ""))

def get_use_megatron():
    return str_to_bool(os.environ.get("ENGINE_USE_MEGATRON", ""))


@contextmanager
def patch_environment(**kwargs):
    """
    A context manager that will add each keyword argument passed to `os.environ` and remove them when exiting.
    """
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
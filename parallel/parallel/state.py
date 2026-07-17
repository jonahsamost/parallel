import random
import numpy as np
import os
from enum import Enum
from typing import Optional
import torch
from dataclasses import dataclass, field
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from omegaconf import DictConfig, OmegaConf

from .utils import is_dist_initialized


class Strategies(str, Enum):
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    TP = "tp"
    CP = "cp"
    SP = "sp"
    EP = "ep"
    PP = "pp"


class RuntimeState:
    _state = {}
    def __init__(self, backend: str | None = None):
        if self.initialized:
            return
        self._state["RANK"] = int(os.environ.get("RANK", 0))
        self._state["WORLD_SIZE"] = int(os.environ.get("WORLD_SIZE", 1))
        self._state["LOCAL_RANK"] = int(os.environ.get("LOCAL_RANK", 0))
        self._state["backend"] = backend
    
    @property
    def can_log(self):
        return self.world_size <= 1 or self.rank == 0

    @property
    def local_rank(self):
        return self._state.get("LOCAL_RANK", 0)
    
    @property
    def rank(self):
        return self._state.get("RANK", 0)
    
    @property
    def world_size(self):
        return self._state.get("WORLD_SIZE", 1)
    
    @property
    def backend(self):
        return self._state.get("backend", None)

    @property
    def initialized(self):
        return self._state != {}    


def init_dist(cfg: DictConfig):
    device = cfg.config.device_type
    assert device, "device needs to be set in config"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if cfg.config.backend == "nccl" and device.type != "cuda":
            raise RuntimeError("nccl needs to be used with cuda devices")
        dist.init_process_group(backend=cfg.config.backend)


@dataclass
class ParallelConfig:
    dp_replicate_size: Optional[int] = None
    dp_shard_size: Optional[int] = None
    tp_size: Optional[int] = None
    cp_size: Optional[int] = None
    sp_size: Optional[int] = None
    ep_size: Optional[int] = None
    pp_size: Optional[int] = None
    device_mesh = None
    device_type = None

    rank: int = -1
    local_rank: int = -1
    world_size: int = -1

    def __init__(self, cfg: DictConfig):
        conf = cfg.parallel
        self.dp_replicate_size = conf.dp_replicate
        self.dp_shard_size = conf.dp_shard
        self.tp_size = conf.tp
        self.cp_size = conf.cp
        self.sp_size = conf.sp
        self.ep_size = conf.ep
        self.pp_size = conf.pp

        self.rank = dist.get_rank() if is_dist_initialized() else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size() if is_dist_initialized() else 1

        self._device = None

        if self.world_size > 1:
            assert is_dist_initialized()

        sizes = {
            "dp_replicate": self.dp_replicate_size,
            "dp_shard": self.dp_shard_size,
            "tp": self.tp_size,
            "cp": self.cp_size,
            "sp": self.sp_size,
            "ep": self.ep_size,
            "pp": self.pp_size,
        }
        invalid = {name: size for name, size in sizes.items() if not isinstance(size, int) or size < 1}
        if invalid:
            raise ValueError(f"Parallel dimensions must be positive integers: {invalid}")
        if self.total_size != self.world_size:
            raise ValueError(
                f"Parallel mesh size {self.total_size} does not match distributed world size {self.world_size}"
            )
        unsupported = {name: size for name, size in sizes.items() if name not in {"dp_replicate", "dp_shard"} and size > 1}
        if unsupported:
            raise NotImplementedError(f"Only replicated and sharded data parallelism are implemented: {unsupported}")
    
    @property
    def dp_size(self):
        return self.dp_replicate_size * self.dp_shard_size
    
    @property
    def dp_enabled(self):
        return self.dp_size > 1
    
    @property
    def is_distributed(self):
        return self.world_size > 1
    
    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def data_rank(self):
        replicate_rank = self._mesh_rank(Strategies.DP_REPLICATE)
        shard_rank = self._mesh_rank(Strategies.DP_SHARD)
        return replicate_rank * self.dp_shard_size + shard_rank

    @property
    def data_world_size(self):
        return self.dp_size

    def __repr__(self):
        return (
            "TopoConfig(\n "
            f"\tdp_replicate_size={self.dp_replicate_size},\n"
            f"\tdp_shard_size={self.dp_shard_size},\n"
            f"\ttp_size={self.tp_size},\n"
            f"\tcp_size={self.cp_size},\n"
            f"\tsp_size={self.sp_size},\n"
            f"\tep_size={self.ep_size},\n"
            f"\tpp_size={self.pp_size},\n"
            f"\ttotal_size={self.total_size}\n"
        )
    
    @property
    def total_size(self):
        return (
            self.dp_replicate_size * self.dp_shard_size
            * self.tp_size * self.cp_size * self.sp_size
            * self.ep_size * self.pp_size
        )
    
    def get_mesh_dims(self):
        dims = [
            (Strategies.DP_REPLICATE, self.dp_replicate_size),
            (Strategies.DP_SHARD, self.dp_shard_size),
            (Strategies.TP, self.tp_size),
            (Strategies.CP, self.cp_size),
            (Strategies.SP, self.sp_size),
            (Strategies.EP, self.ep_size),
            (Strategies.PP, self.pp_size),
        ]
        dims = [x for x in dims if x[1] > 1]
        return tuple(zip(*dims))
    
    def set_device_mesh(self, device_type: str):
        self.device_type = device_type
        dims = self.get_mesh_dims()
        if len(dims) == 0:
            self.device_mesh = None
            return None
        mesh_dim_names, mesh_shape = dims
        device_mesh = init_device_mesh(
            device_type,
            mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )
        self.device_mesh = device_mesh
        return self.device_mesh
    
    @property
    def device(self):
        if self._device is not None:
            return self._device
        if self.device_type == "cuda":
            self._device = torch.device("cuda", self.local_rank)
        else:
            self._device = torch.device("cpu")
        return self._device

    def _mesh_rank(self, dim_name: str) -> int:
        if self.device_mesh is not None and dim_name in self.device_mesh.mesh_dim_names:
            return self.device_mesh.get_local_rank(dim_name)
        return 0

    def model_init_rngs(self, seed: int = 42):
        """
        different rank TP get different seeds
        """
        tp_rank = self._mesh_rank(Strategies.TP)
        init_seed = seed + tp_rank
        torch.manual_seed(init_seed)
        if self.device_type == "cuda":
            torch.cuda.manual_seed_all(init_seed)

    def model_train_rngs(self, seed: int = 42):
        """
        - DP ranks get different seeds (each sees different data)
        - TP ranks within same (DP/PP) group get same seed
        """
        dp_rank = self.data_rank
        pp_rank = self._mesh_rank(Strategies.PP)

        data_seed = seed + dp_rank
        random.seed(data_seed)
        np.random.seed(data_seed)

        torch_seed = seed + dp_rank * 1000 + pp_rank
        torch.manual_seed(torch_seed)
        if self.device_type == "cuda":
            torch.cuda.manual_seed_all(torch_seed)

import os
from typing import Optional
from parallel.parallel.utils import is_cuda_available
import torch
from dataclasses import dataclass, field
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from omegaconf import DictConfig, OmegaConf


class RuntimeState:
    _state = {}
    def __init__(self, backend: str | None = None):
        if self.initialized:
            return
        self._state["RANK"] = int(os.environ.get("RANK", 0))
        self._state["WORLD_SIZE"] = int(os.environ.get("WORLD_SIZE", 0))
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
        return self._state.get("WORLD_SIZE", 0)
    
    @property
    def backend(self):
        return self._state.get("backend", None)

    def initialized(self):
        return self._state != {}    


def init_dist(cfg: DictConfig):
    device = cfg.config.device_type
    assert device, "device needs to be set in config"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device == "cuda":
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

        self.rank = dist.get_rank()
        self.local_rank = self.rank % 8
        self.world_size = dist.get_world_size()
    
    def is_distributed(self):
        return self.world_size > 1
    
    def is_main_process(self):
        return self.rank == 0

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
            ("dp_replicate", self.dp_replicate_size),
            ("dp_shard", self.dp_shard_size),
            ("tp", self.tp_size),
            ("cp", self.cp_size),
            ("sp", self.sp_size),
            ("ep", self.ep_size),
            ("pp", self.pp_size),
        ]
        dims = [x for x in dims if x[1] > 1]
        return tuple(zip(*dims))
    
    def set_device_mesh(self, device_type: str):
        dims = self.get_mesh_dims()
        assert len(dims) > 0, "Mesh dims length == 0"
        mesh_dim_names, mesh_shape = dims
        device_mesh = init_device_mesh(
            device_type,
            mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )
        self.device_mesh = device_mesh
        self.device_type = device_type
        return self.device_mesh
    


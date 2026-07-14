from typing import Optional
from parallel.parallel.utils import is_cuda_available
import torch
from dataclasses import dataclass, field
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


@dataclass
class TopoConfig:



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

    rank: int = -1
    local_rank: int = -1
    world_size: int = -1

    def __post_init__(self):
        self.rank = dist.get_rank()
        self.local_rank = self.rank % 8
        self.world_size = dist.get_world_size()

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
            ("dp_replicate", self.dp),
            ("dp_shard", self.dp),
            ("tp", self.tp),
            ("cp", self.cp),
            ("sp", self.sp),
            ("ep", self.ep),
            ("pp", self.pp),
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
        return self.device_mesh
    
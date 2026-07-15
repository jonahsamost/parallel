import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from parallel.state import ParallelConfig, Strategies
from parallel.utils import is_dist_initialized, empty_fn


def communicate_rank_loss(loss: torch.Tensor, pconfig: ParallelConfig):
    if not pconfig.dp_enabled or pconfig.device_mesh is None:
        return
    dims = [
        pconfig.device_mesh.get_group(d)
        for d in (Strategies.DP_REPLICATE, Strategies.DP_SHARD)
        if d in pconfig.device_mesh.mesh_dim_names
    ]
    for dp_group in dims:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=dp_group)


def average_gradients(model: AutoModelForCausalLM, pconfig: ParallelConfig):
    if pconfig.dp_replicate_size <= 1 or pconfig.device_mesh is None:
        return
    dp_replicate_group = pconfig.device_mesh.get_group(Strategies.DP_REPLICATE)
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=dp_replicate_group)

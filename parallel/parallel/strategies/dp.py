import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from parallel.state import ParallelConfig, Strategies
from parallel.utils import is_dist_initialized, empty_fn


def communicate_rank_loss(loss: torch.Tensor, pconfig: ParallelConfig):
    if pconfig.dp_enabled:
        dims = [
            (d, pconfig.device_mesh.group(d))
            for d in (Strategies.DP_REPLICATE, Strategies.DP_SHARD)
            if d in pconfig.device_mesh.mesh_dim_names
        ]
        for _, dp_group in dims:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=dp_group)


def average_gradients(model: AutoModelForCausalLM, pconfig: ParallelConfig):
    if pconfig.dp_replicate_size > 1:
        dp_replicate_group = pconfig.device_mesh.group(Strategies.DP_REPLICATE)
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=dp_replicate_group)

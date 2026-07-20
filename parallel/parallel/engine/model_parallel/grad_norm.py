from __future__ import annotations

import torch
import torch.distributed as dist

from ...state import Strategies
from .plan import ModelParallelPlan


def _local_tensor(tensor):
    return tensor._local_tensor if hasattr(tensor, "_local_tensor") else tensor


def global_grad_norm(
    named_parameters,
    *,
    plan: ModelParallelPlan,
    tp_mesh,
    pconfig,
    device,
) -> torch.Tensor:
    """Compute one global norm without double-counting replicated parameters."""
    local_sq_norm = torch.zeros((), dtype=torch.float32, device=device)
    tp_rank = tp_mesh.get_local_rank()
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        placement = plan.placement_for_parameter(name)
        if placement.shard_dim is None and tp_rank != 0:
            continue
        gradient = _local_tensor(parameter.grad)
        local_sq_norm += gradient.detach().float().pow(2).sum().to(device)
    dist.all_reduce(local_sq_norm, op=dist.ReduceOp.SUM, group=tp_mesh.get_group())
    mesh = pconfig.device_mesh
    if pconfig.dp_shard_size > 1:
        dp_shard_group = mesh.get_group(Strategies.DP_SHARD)
        dist.all_reduce(
            local_sq_norm,
            op=dist.ReduceOp.SUM,
            group=dp_shard_group,
        )
    return local_sq_norm.sqrt()


def clip_grad_norm_(
    named_parameters,
    max_norm: float,
    *,
    plan: ModelParallelPlan,
    tp_mesh,
    pconfig,
    device,
) -> torch.Tensor:
    named_parameters = tuple(named_parameters)
    total_norm = global_grad_norm(
        named_parameters,
        plan=plan,
        tp_mesh=tp_mesh,
        pconfig=pconfig,
        device=device,
    )
    coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for _, parameter in named_parameters:
        if parameter.grad is not None:
            gradient = _local_tensor(parameter.grad)
            gradient.mul_(coefficient.to(gradient.device))
    return total_norm

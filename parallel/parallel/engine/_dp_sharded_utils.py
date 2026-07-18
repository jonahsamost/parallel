from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor


def group_named_parameters(
    named_parameters: list[tuple[str, nn.Parameter]],
) -> list[list[tuple[str, nn.Parameter]]]:
    groups: dict[
        tuple[torch.device, torch.dtype, bool],
        list[tuple[str, nn.Parameter]],
    ] = {}
    for name, parameter in named_parameters:
        key = (parameter.device, parameter.dtype, parameter.requires_grad)
        groups.setdefault(key, []).append((name, parameter))
    return list(groups.values())


def get_fsdp_units(model: nn.Module) -> list[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    return [model]


def discover_used_parameter_ids(
    loss: torch.Tensor,
    parameter_ids: set[int],
) -> set[int]:
    """Find leaf parameters reachable from this loss's autograd graph."""
    used_parameter_ids: set[int] = set()
    if id(loss) in parameter_ids:
        used_parameter_ids.add(id(loss))
    if loss.grad_fn is None:
        return used_parameter_ids

    stack = [loss.grad_fn]
    visited_nodes: set[object] = set()
    while stack:
        node = stack.pop()
        if node in visited_nodes:
            continue
        visited_nodes.add(node)

        variable = getattr(node, "variable", None)
        if variable is not None and id(variable) in parameter_ids:
            used_parameter_ids.add(id(variable))
        for next_node, _ in node.next_functions:
            if next_node is not None:
                stack.append(next_node)
    return used_parameter_ids


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor

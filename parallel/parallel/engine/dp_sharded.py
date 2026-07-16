from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..state import ParallelConfig, Strategies
from ._dp_param_unit import DPParamUnit


def _group_named_parameters(
    named_parameters: list[tuple[str, nn.Parameter]],
) -> list[list[tuple[str, nn.Parameter]]]:
    groups: dict[tuple[torch.device, torch.dtype, bool], list[tuple[str, nn.Parameter]]] = {}
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


class FSDPWrapper:
    """ Wraps a model for FSDP """

    def __init__(
        self,
        model: nn.Module,
        pconfig: ParallelConfig,
        cpu_offload: bool = False,
        activation_checkpoint: bool = False,
        checkpoint_every_n: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.pconfig = pconfig
        self.cpu_offload = cpu_offload
        self.activation_checkpoint = activation_checkpoint
        self.checkpoint_every_n = max(checkpoint_every_n, 1)
        self.device = device

        self.units: list[DPParamUnit] = []
        self.dp_shard_group: Optional[dist.ProcessGroup] = None

    def shard_model(self):
        """Walk the model, create DPParamUnits, shard params, wire prefetch."""
        if self.pconfig.dp_shard_size <= 1 or self.pconfig.device_mesh is None:
            return
        if self.units:
            raise RuntimeError("FSDPWrapper.shard_model() can only be called once")

        self.dp_shard_group = self.pconfig.device_mesh.get_group(Strategies.DP_SHARD)
        if self.device is not None:
            for buffer in self.model.buffers():
                buffer.data = buffer.data.to(self.device)
        submodules = get_fsdp_units(self.model)
        unit_specs: list[tuple[nn.Module, list[tuple[str, nn.Parameter]], bool, bool]] = []
        owned_param_ids: set[int] = set()

        for i, submodule in enumerate(submodules):
            named_params = list(submodule.named_parameters())
            duplicate_ids = owned_param_ids.intersection(id(param) for _, param in named_params)
            if duplicate_ids:
                raise ValueError("A parameter is shared by multiple FSDP units")
            if not named_params:
                continue
            owned_param_ids.update(id(param) for _, param in named_params)
            checkpoint_module = self.activation_checkpoint and (i % self.checkpoint_every_n == 0)
            reshard_after_forward = submodule is not self.model
            for group_index, param_group in enumerate(_group_named_parameters(named_params)):
                unit_specs.append(
                    (
                        submodule,
                        param_group,
                        checkpoint_module and group_index == 0,
                        reshard_after_forward,
                    )
                )

        if submodules != [self.model]:
            root_params = [
                (name, param)
                for name, param in self.model.named_parameters()
                if id(param) not in owned_param_ids
            ]
            if root_params:
                root_specs = [
                    (self.model, param_group, False, False)
                    for param_group in _group_named_parameters(root_params)
                ]
                unit_specs[0:0] = root_specs
                owned_param_ids.update(id(param) for _, param in root_params)

        model_param_ids = {id(param) for param in self.model.parameters()}
        if owned_param_ids != model_param_ids:
            raise RuntimeError("Every model parameter must belong to exactly one FSDP unit")

        for i, (submodule, named_params, use_ckpt, reshard_after_forward) in enumerate(unit_specs):
            unit = DPParamUnit(
                module=submodule,
                dp_shard_group=self.dp_shard_group,
                unit_index=i,
                cpu_offload=self.cpu_offload,
                use_activation_checkpoint=use_ckpt,
                device=self.device,
                named_parameters=named_params,
                reshard_after_forward=reshard_after_forward,
            )
            unit.shard()
            self.units.append(unit)

        self._wire_prefetch()

    def _wire_prefetch(self):
        """Link each unit to the next unit for forward/backward prefetch."""
        for i, unit in enumerate(self.units):
            if i + 1 < len(self.units):
                unit.next_fwd = self.units[i + 1]
            if i - 1 >= 0:
                unit.next_bwd = self.units[i - 1]

    def get_optimizer_params(self) -> list[nn.Parameter]:
        if not self.units:
            return [param for param in self.model.parameters() if param.requires_grad]
        return [
            unit.flat_shard
            for unit in self.units
            if unit.requires_grad and unit.flat_shard is not None
        ]

    def get_sharded_params(self) -> list[nn.Parameter]:
        return [unit.flat_shard for unit in self.units if unit.flat_shard is not None]

    def finalize_backward(self):
        """Run reductions only after Tensor.backward() has completed."""
        for unit in reversed(self.units):
            unit.finalize_backward()

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        """Clip using one norm shared by every rank in the shard group."""
        params = self.get_optimizer_params()
        if not self.is_active:
            return torch.nn.utils.clip_grad_norm_(params, max_norm)

        local_sq_norm = torch.zeros((), dtype=torch.float32, device=self.device)
        for param in params:
            if param.grad is not None:
                local_sq_norm += param.grad.detach().float().pow(2).sum().to(self.device)
        dist.all_reduce(local_sq_norm, op=dist.ReduceOp.SUM, group=self.dp_shard_group)
        total_norm = local_sq_norm.sqrt()
        clip_coefficient = max_norm / (total_norm + 1e-6)
        if clip_coefficient < 1:
            for param in params:
                if param.grad is not None:
                    param.grad.mul_(clip_coefficient.to(param.grad.device))
        return total_norm

    def state_dict(self) -> OrderedDict:
        """Return a full, detached state dict on every shard rank."""
        if not self.is_active:
            return self.model.state_dict()
        for unit in self.units:
            unit.all_gather()
        try:
            model_state = self.model.state_dict()
            state = OrderedDict(
                (
                    name,
                    value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value),
                )
                for name, value in model_state.items()
            )
            if hasattr(model_state, "_metadata"):
                state._metadata = copy.deepcopy(model_state._metadata)
            return state
        finally:
            for unit in self.units:
                unit.free_full_params()

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load a full state dict and write each rank's optimizer shards."""
        if not self.is_active:
            return self.model.load_state_dict(state_dict, strict=strict)
        for unit in self.units:
            unit.all_gather()
        try:
            result = self.model.load_state_dict(state_dict, strict=strict)
            for unit in self.units:
                unit.writeback_shard()
            return result
        finally:
            for unit in self.units:
                unit.free_full_params()

    @property
    def is_active(self) -> bool:
        return len(self.units) > 0

    def remove_hooks(self):
        for unit in self.units:
            unit.all_gather()
        for unit in self.units:
            unit.remove_hooks()

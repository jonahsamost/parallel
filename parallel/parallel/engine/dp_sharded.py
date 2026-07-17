from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..state import ParallelConfig, Strategies
from ._dp_param_unit import DPParamUnit
from ._dp_sharded_utils import (
    discover_used_parameter_ids,
    get_fsdp_units,
    group_named_parameters,
)


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
        self._backward_units: list[DPParamUnit] = []
        self._backward_positions: dict[int, int] = {}
        self._ready_backward_positions: set[int] = set()
        self._next_backward_position: int = -1
        self._inflight_backward_positions: list[int] = []
        self._max_inflight_gradient_reductions: int = 2
        self._backward_prepared: bool = False

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
            for group_index, param_group in enumerate(group_named_parameters(named_params)):
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
                    for param_group in group_named_parameters(root_params)
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

        self._initialize_backward_reducer()
        self._wire_prefetch()

    def _initialize_backward_reducer(self):
        """Build the fixed collective schedule shared by all shard ranks."""
        self._backward_units = [unit for unit in self.units if unit.requires_grad]
        self._backward_positions = {
            id(unit): position for position, unit in enumerate(self._backward_units)
        }
        self._ready_backward_positions.clear()
        self._next_backward_position = len(self._backward_units) - 1
        self._inflight_backward_positions.clear()
        for unit in self._backward_units:
            unit.set_backward_ready_callback(self._on_unit_gradients_ready)

    def prepare_backward(self, loss: torch.Tensor):
        """Discover this graph's used parameters and pre-mark empty units ready."""
        if not self.is_active:
            return
        if not isinstance(loss, torch.Tensor):
            raise TypeError("FSDPWrapper.prepare_backward() requires a loss tensor")
        if self._backward_prepared:
            raise RuntimeError("FSDP backward is already prepared")

        parameter_ids = {
            parameter_id
            for unit in self._backward_units
            for parameter_id in unit.parameter_ids
        }
        used_parameter_ids = discover_used_parameter_ids(loss, parameter_ids)
        zero_ready_positions: set[int] = set()
        for position, unit in enumerate(self._backward_units):
            if unit.prepare_backward(used_parameter_ids):
                zero_ready_positions.add(position)

        self._ready_backward_positions.update(zero_ready_positions)
        self._backward_prepared = True

    def _on_unit_gradients_ready(self, unit: DPParamUnit):
        """Record local readiness and drain only the next globally ordered units."""
        if not self._backward_prepared:
            raise RuntimeError("FSDP unit became ready before prepare_backward()")
        try:
            position = self._backward_positions[id(unit)]
        except KeyError as error:
            raise RuntimeError("Unknown FSDP unit reported gradient readiness") from error
        self._ready_backward_positions.add(position)
        self._drain_backward_reductions()

    def _drain_backward_reductions(self):
        while self._next_backward_position in self._ready_backward_positions:
            position = self._next_backward_position
            self._ready_backward_positions.remove(position)
            while (
                len(self._inflight_backward_positions)
                >= self._max_inflight_gradient_reductions
            ):
                oldest_position = self._inflight_backward_positions.pop(0)
                self._backward_units[oldest_position].finish_gradient_reduction()
            self._backward_units[position].start_gradient_reduction()
            self._inflight_backward_positions.append(position)
            self._next_backward_position -= 1

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
        """Finish bounded in-flight reductions and commit optimizer gradients."""
        if not self.is_active:
            return
        if not self._backward_prepared:
            raise RuntimeError("FSDP backward was not prepared before finalization")

        # This is needed when the entire local model was unused and no parameter
        # hook fired to kick the precomputed zero-ready schedule.
        self._drain_backward_reductions()
        if self._next_backward_position != -1:
            pending = [
                {
                    "unit": unit.unit_index,
                    "missing_gradients": unit.missing_expected_grad_names(),
                }
                for unit in self._backward_units
                if not unit.backward_ready
            ]
            raise RuntimeError(
                "FSDP backward ended before every expected parameter gradient "
                f"was accumulated: {pending}"
            )

        if self._backward_units:
            for position in self._inflight_backward_positions:
                self._backward_units[position].finish_gradient_reduction()
            self._inflight_backward_positions.clear()

            participated = torch.tensor(
                [int(unit.local_gradient_participated) for unit in self._backward_units],
                dtype=torch.int32,
                device=self._backward_units[0].device,
            )
            dist.all_reduce(participated, op=dist.ReduceOp.MAX, group=self.dp_shard_group)
            for position in range(len(self._backward_units) - 1, -1, -1):
                self._backward_units[position].accumulate_reduced_gradient(
                    accumulate=bool(participated[position].item())
                )
        for unit in self.units:
            if not unit.requires_grad:
                unit.free_full_params()
            unit.reset_backward_state()

        self._ready_backward_positions.clear()
        self._next_backward_position = len(self._backward_units) - 1
        self._backward_prepared = False

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

    # TODO dont gather entire model onto one device
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

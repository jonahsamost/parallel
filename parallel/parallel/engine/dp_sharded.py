from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from ..state import ParallelConfig, Strategies
from ._dp_param_unit import DPParamUnit
from ._dp_sharded_utils import (
    discover_used_parameter_ids,
    get_fsdp_units,
    group_named_parameters,
    local_tensor,
)


class FSDPWrapper:
    """ Wraps a model for FSDP 
    
    Note on find_unused_parameters:

    when false -> the normal path (for dense models). The engine assumes that every trainable
    parameter receives a gradient. Every DP-shard rank uses the same FSDP units. 
        all-gather layer N parameters
        compute layer N gradients
        reduce-scatter layer N gradients
        free temporary full parameters and gradients
        continue to layer N-1
    when true -> inspect which parameters each rank uses and coordinate comms when
        their backward graphs differ

    really we want to know if any trainable parameter can have `param.grad is None` on this step,
        especially on only some ranks. 
    
    """

    def __init__(
        self,
        model: nn.Module,
        pconfig: ParallelConfig,
        cpu_offload: bool = False,
        activation_checkpoint: bool = False,
        checkpoint_every_n: int = 1,
        find_unused_parameters: bool = False,
        overlap_backward_reductions: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.pconfig = pconfig
        self.cpu_offload = cpu_offload
        self.activation_checkpoint = activation_checkpoint
        self.checkpoint_every_n = max(checkpoint_every_n, 1)
        self.find_unused_parameters = find_unused_parameters
        self.overlap_backward_reductions = overlap_backward_reductions
        first_parameter = next(model.parameters(), None)
        self.device = device or (
            first_parameter.device if first_parameter is not None else torch.device("cpu")
        )

        self.units: list[DPParamUnit] = []
        self.dp_shard_group: Optional[dist.ProcessGroup] = None
        self.dp_shard_mesh = None
        self._backward_units: list[DPParamUnit] = []
        self._backward_positions: dict[int, int] = {}
        self._ready_backward_positions: set[int] = set()
        self._next_backward_position: int = -1
        self._inflight_backward_positions: list[int] = []
        self._max_inflight_gradient_reductions: int = 2
        self._backward_prepared: bool = False
        self._globally_used_backward_positions: list[bool] = []
        self._global_parameter_participation: dict[int, list[bool]] = {}
        self._deferred_backward_positions: list[int] = []
        self._reduced_backward_positions: set[int] = set()
        self._state_dict_keys: tuple[str, ...] = ()
        self._parameter_state_names: dict[int, tuple[str, ...]] = {}
        self._parameter_state_keys: set[str] = set()

    def _capture_state_dict_layout(self) -> None:
        """Capture canonical model keys before parameters become placeholders."""
        state = self.model.state_dict(keep_vars=True)
        parameter_ids = {id(parameter) for parameter in self.model.parameters()}
        names_by_parameter: dict[int, list[str]] = {
            parameter_id: [] for parameter_id in parameter_ids
        }
        for name, value in state.items():
            if id(value) in names_by_parameter:
                names_by_parameter[id(value)].append(name)

        missing = [
            name
            for name, parameter in self.model.named_parameters()
            if not names_by_parameter[id(parameter)]
        ]
        if missing:
            raise RuntimeError(
                "Every model parameter must be represented in model.state_dict(); "
                f"missing parameters: {missing}"
            )

        self._state_dict_keys = tuple(state.keys())
        self._parameter_state_names = {
            parameter_id: tuple(names)
            for parameter_id, names in names_by_parameter.items()
        }
        self._parameter_state_keys = {
            name
            for names in self._parameter_state_names.values()
            for name in names
        }

    def shard_model(self):
        """Walk the model, create DPParamUnits, shard params, wire prefetch."""
        self._capture_state_dict_layout()
        if self.pconfig.dp_shard_size <= 1 or self.pconfig.device_mesh is None:
            return
        if self.units:
            raise RuntimeError("FSDPWrapper.shard_model() can only be called once")

        self.dp_shard_mesh = self.pconfig.device_mesh[Strategies.DP_SHARD]
        self.dp_shard_group = self.dp_shard_mesh.get_group()
        if self.device is not None:
            for buffer in self.model.buffers():
                buffer.data = buffer.data.to(self.device)
        submodules = get_fsdp_units(self.model)
        unit_specs: list[tuple[nn.Module, list[tuple[str, nn.Parameter]], bool, bool]] = []
        owned_param_ids: set[int] = set()

        for i, submodule in enumerate(submodules):
            named_params = [
                (self._parameter_state_names[id(parameter)][0], parameter)
                for _, parameter in submodule.named_parameters()
            ]
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
                (self._parameter_state_names[id(param)][0], param)
                for _, param in self.model.named_parameters()
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
                dp_shard_mesh=self.dp_shard_mesh,
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
        self._deferred_backward_positions.clear()
        self._reduced_backward_positions.clear()
        for unit in self._backward_units:
            unit.set_backward_ready_callback(self._on_unit_gradients_ready)

    def prepare_backward(self, loss: torch.Tensor):
        """Prepare either the fast or usage-aware backward schedule.

        The fast path assumes every rank reaches the same trainable parameters
        in compatible reverse-unit order. The usage-aware path discovers the
        local autograd graph, sums one combined unit/parameter bitmap across
        the shard group, and lets the wrapper issue just-in-time all-gathers on
        every rank. A locally unused rank therefore still contributes its
        parameter shard and later supplies zero gradients for a globally used
        unit without materializing all units before backward.
        """
        if not self.is_active:
            return
        if not isinstance(loss, torch.Tensor):
            raise TypeError("FSDPWrapper.prepare_backward() requires a loss tensor")
        if self._backward_prepared:
            raise RuntimeError("FSDP backward is already prepared")

        self._ready_backward_positions.clear()
        self._deferred_backward_positions.clear()
        self._reduced_backward_positions.clear()
        self._global_parameter_participation.clear()
        self._next_backward_position = len(self._backward_units) - 1

        if not self.find_unused_parameters:
            self._globally_used_backward_positions = [
                True for _ in self._backward_units
            ]
            for position, unit in enumerate(self._backward_units):
                unit.managed_backward_unshard = False
                if unit.prepare_backward(None):
                    self._ready_backward_positions.add(position)
            self._backward_prepared = True
            return

        parameter_ids = {
            parameter_id
            for unit in self._backward_units
            for parameter_id in unit.parameter_ids
        }
        used_parameter_ids = discover_used_parameter_ids(loss, parameter_ids)
        local_unit_usage = [
            int(bool(unit.parameter_ids.intersection(used_parameter_ids)))
            for unit in self._backward_units
        ]
        local_parameter_usage = [
            int(id(meta.parameter) in used_parameter_ids)
            for unit in self._backward_units
            for meta in unit.param_metas
        ]
        usage_counts = torch.tensor(
            local_unit_usage + local_parameter_usage,
            dtype=torch.int32,
            device=self.device,
        )
        if usage_counts.numel():
            dist.all_reduce(
                usage_counts,
                op=dist.ReduceOp.SUM,
                group=self.dp_shard_group,
            )

        unit_count = len(self._backward_units)
        self._globally_used_backward_positions = [
            bool(value)
            for value in usage_counts[:unit_count].gt(0).tolist()
        ]
        parameter_counts = usage_counts[unit_count:]
        parameter_offset = 0
        for position, unit in enumerate(self._backward_units):
            parameter_count = len(unit.param_metas)
            self._global_parameter_participation[position] = [
                bool(value)
                for value in parameter_counts[
                    parameter_offset : parameter_offset + parameter_count
                ].gt(0).tolist()
            ]
            parameter_offset += parameter_count
            unit.managed_backward_unshard = True
            if unit.prepare_backward(used_parameter_ids):
                self._ready_backward_positions.add(position)

        self._backward_prepared = True
        self._advance_usage_aware_backward()

    def _on_unit_gradients_ready(self, unit: DPParamUnit):
        """Record local readiness and advance the configured global schedule."""
        if not self._backward_prepared:
            raise RuntimeError("FSDP unit became ready before prepare_backward()")
        try:
            position = self._backward_positions[id(unit)]
        except KeyError as error:
            raise RuntimeError("Unknown FSDP unit reported gradient readiness") from error
        self._ready_backward_positions.add(position)
        if self.find_unused_parameters:
            self._advance_usage_aware_backward()
        elif self.overlap_backward_reductions:
            self._drain_backward_reductions()

    def _launch_gradient_reduction(self, position: int) -> None:
        while (
            len(self._inflight_backward_positions)
            >= self._max_inflight_gradient_reductions
        ):
            oldest_position = self._inflight_backward_positions.pop(0)
            self._backward_units[oldest_position].finish_gradient_reduction()
        self._backward_units[position].start_gradient_reduction()
        self._inflight_backward_positions.append(position)
        self._reduced_backward_positions.add(position)

    def _drain_backward_reductions(self):
        while self._next_backward_position in self._ready_backward_positions:
            position = self._next_backward_position
            self._ready_backward_positions.remove(position)
            self._launch_gradient_reduction(position)
            self._next_backward_position -= 1

    def _advance_usage_aware_backward(self) -> None:
        """Advance the globally ordered just-in-time backward schedule.

        Each globally used unit is unsharded before this rank either waits for
        its real gradients or immediately contributes zeros. With overlap, the
        per-rank collective order is ``AG(unit), RS(unit), AG(previous), ...``.
        Without overlap, all just-in-time all-gathers happen during backward
        and the fixed-order reduce-scatters are launched during finalization.
        """
        while self._next_backward_position >= 0:
            position = self._next_backward_position
            if not self._globally_used_backward_positions[position]:
                self._next_backward_position -= 1
                continue

            unit = self._backward_units[position]
            unit.schedule_backward_unshard()
            if position not in self._ready_backward_positions:
                return

            self._ready_backward_positions.remove(position)
            if self.overlap_backward_reductions:
                self._launch_gradient_reduction(position)
            else:
                unit.free_full_params()
                self._deferred_backward_positions.append(position)
            self._next_backward_position -= 1

    def _launch_deferred_backward_reductions(self) -> None:
        positions = (
            self._deferred_backward_positions
            if self.find_unused_parameters
            else list(range(len(self._backward_units) - 1, -1, -1))
        )
        for position in positions:
            self._launch_gradient_reduction(position)

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
            meta.sharded_param
            for unit in self.units
            if unit.requires_grad
            for meta in unit.param_metas
            if meta.sharded_param is not None
        ]

    def get_sharded_params(self) -> list[nn.Parameter]:
        return [
            meta.sharded_param
            for unit in self.units
            for meta in unit.param_metas
            if meta.sharded_param is not None
        ]

    def finalize_backward(self):
        """Finish bounded in-flight reductions and commit optimizer gradients."""
        if not self.is_active:
            return
        if not self._backward_prepared:
            raise RuntimeError("FSDP backward was not prepared before finalization")

        if self.find_unused_parameters:
            self._advance_usage_aware_backward()
        elif self.overlap_backward_reductions:
            self._drain_backward_reductions()
        else:
            missing_positions = [
                position
                for position, unit in enumerate(self._backward_units)
                if not unit.backward_ready
            ]
            if not missing_positions:
                self._next_backward_position = -1

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

        if not self.overlap_backward_reductions:
            self._launch_deferred_backward_reductions()

        if self._backward_units:
            for position in self._inflight_backward_positions:
                self._backward_units[position].finish_gradient_reduction()
            self._inflight_backward_positions.clear()

            if not self.find_unused_parameters:
                for position, unit in enumerate(self._backward_units):
                    self._global_parameter_participation[position] = [
                        True for _ in unit.param_metas
                    ]

            for position in range(len(self._backward_units) - 1, -1, -1):
                if position not in self._reduced_backward_positions:
                    continue
                self._backward_units[position].accumulate_reduced_gradients(
                    self._global_parameter_participation[position]
                )
        for unit in self.units:
            unit.free_full_params()
            unit.reset_backward_state()

        self._ready_backward_positions.clear()
        self._next_backward_position = len(self._backward_units) - 1
        self._backward_prepared = False
        self._globally_used_backward_positions.clear()
        self._global_parameter_participation.clear()
        self._deferred_backward_positions.clear()
        self._reduced_backward_positions.clear()

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        """Clip using one norm shared by every rank in the shard group."""
        params = self.get_optimizer_params()
        if not self.is_active:
            return torch.nn.utils.clip_grad_norm_(params, max_norm)

        local_sq_norm = torch.zeros((), dtype=torch.float32, device=self.device)
        for param in params:
            if param.grad is not None:
                grad = local_tensor(param.grad)
                local_sq_norm += grad.detach().float().pow(2).sum().to(self.device)
        dist.all_reduce(local_sq_norm, op=dist.ReduceOp.SUM, group=self.dp_shard_group)
        total_norm = local_sq_norm.sqrt()
        clip_coefficient = max_norm / (total_norm + 1e-6)
        if clip_coefficient < 1:
            for param in params:
                if param.grad is not None:
                    grad = local_tensor(param.grad)
                    grad.mul_(clip_coefficient.to(grad.device))
        return total_norm

    @staticmethod
    def _cpu_copy(value: Any) -> Any:
        if isinstance(value, DTensor):
            return value._local_tensor.detach().to(device="cpu", copy=True)
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", copy=True)
        return copy.deepcopy(value)

    def _replicate_rank(self) -> int:
        mesh = self.pconfig.device_mesh
        if (
            mesh is not None
            and Strategies.DP_REPLICATE in getattr(mesh, "mesh_dim_names", ())
        ):
            return mesh.get_local_rank(Strategies.DP_REPLICATE)
        return 0

    def _is_checkpoint_replica(self) -> bool:
        return self._replicate_rank() == 0

    def _full_state_root(self) -> bool:
        if not self._is_checkpoint_replica():
            return False
        if self.is_active:
            return dist.get_rank(self.dp_shard_group) == 0
        return not dist.is_initialized() or dist.get_rank() == 0

    def _non_parameter_state_dict(self) -> OrderedDict:
        current_state = self.model.state_dict(keep_vars=True)
        state = OrderedDict(
            (name, self._cpu_copy(value))
            for name, value in current_state.items()
            if name not in self._parameter_state_keys
        )
        if hasattr(current_state, "_metadata"):
            state._metadata = copy.deepcopy(current_state._metadata)
        return state

    def _copy_non_parameter_state(self, state_dict) -> None:
        current_state = self.model.state_dict(keep_vars=True)
        for name, destination in current_state.items():
            if name in self._parameter_state_keys or name not in state_dict:
                continue
            source = state_dict[name]
            if isinstance(destination, torch.Tensor):
                destination.detach().copy_(source)
                continue
            suffix = "._extra_state"
            if name == "_extra_state":
                module = self.model
            elif name.endswith(suffix):
                module = self.model.get_submodule(name[: -len(suffix)])
            else:
                continue
            module.set_extra_state(copy.deepcopy(source))

    def _sync_tensor(self, tensor: torch.Tensor, group) -> None:
        source_rank = dist.get_global_rank(group, 0)
        value = local_tensor(tensor).detach()
        if value.device.type == self.device.type:
            dist.broadcast(value, src=source_rank, group=group)
            return
        staged = value.to(self.device)
        dist.broadcast(staged, src=source_rank, group=group)
        value.copy_(staged.to(value.device))

    def _sync_replicated_state(self) -> None:
        mesh = self.pconfig.device_mesh
        if (
            mesh is None
            or Strategies.DP_REPLICATE not in getattr(mesh, "mesh_dim_names", ())
        ):
            return
        group = mesh.get_group(Strategies.DP_REPLICATE)
        tensors = list(self.model.buffers())
        if self.is_active:
            tensors.extend(self.get_sharded_params())
        else:
            tensors.extend(self.model.parameters())
        for tensor in tensors:
            self._sync_tensor(tensor, group)

    def full_state_dict(self) -> Optional[OrderedDict]:
        """Build a full CPU state dict on global rank zero, one unit at a time."""
        if not self._state_dict_keys:
            self._capture_state_dict_layout()
        if not self._is_checkpoint_replica():
            return None
        if not self.is_active:
            if not self._full_state_root():
                return None
            model_state = self.model.state_dict(keep_vars=True)
            state = OrderedDict(
                (name, self._cpu_copy(value)) for name, value in model_state.items()
            )
            if hasattr(model_state, "_metadata"):
                state._metadata = copy.deepcopy(model_state._metadata)
            return state

        root = self._full_state_root()
        parameter_values: dict[str, torch.Tensor] = {}
        for unit in self.units:
            unit.all_gather()
            try:
                if root:
                    for meta in unit.param_metas:
                        cpu_value = self._cpu_copy(meta.parameter)
                        for name in self._parameter_state_names[id(meta.parameter)]:
                            parameter_values[name] = cpu_value
            finally:
                unit.free_full_params()

        if not root:
            return None
        non_parameter_state = self._non_parameter_state_dict()
        state = OrderedDict()
        for name in self._state_dict_keys:
            if name in parameter_values:
                state[name] = parameter_values[name]
            else:
                state[name] = non_parameter_state[name]
        if hasattr(non_parameter_state, "_metadata"):
            state._metadata = copy.deepcopy(non_parameter_state._metadata)
        return state

    def _validate_full_state_dict(self, state_dict, strict: bool) -> dict[str, Any]:
        expected = set(self._state_dict_keys)
        actual = set(state_dict.keys())
        missing = [name for name in self._state_dict_keys if name not in actual]
        unexpected = [name for name in state_dict if name not in expected]

        expected_shapes: dict[str, torch.Size] = {}
        for unit in self.units:
            for meta in unit.param_metas:
                for name in self._parameter_state_names[id(meta.parameter)]:
                    expected_shapes[name] = meta.shape
        current_state = self.model.state_dict(keep_vars=True)
        for name, value in current_state.items():
            if isinstance(value, torch.Tensor) and name not in expected_shapes:
                expected_shapes[name] = value.shape

        errors = []
        if strict and missing:
            errors.append(f"Missing key(s): {missing}")
        if strict and unexpected:
            errors.append(f"Unexpected key(s): {unexpected}")
        for name in expected.intersection(actual):
            value = state_dict[name]
            if name in expected_shapes:
                if not isinstance(value, torch.Tensor):
                    errors.append(f"Expected tensor for {name}, got {type(value).__name__}")
                elif value.shape != expected_shapes[name]:
                    errors.append(
                        f"Shape mismatch for {name}: expected {tuple(expected_shapes[name])}, "
                        f"got {tuple(value.shape)}"
                    )
        return {
            "error": "; ".join(errors) if errors else None,
            "missing": missing,
            "unexpected": unexpected,
        }

    def load_full_state_dict(self, state_dict, strict: bool = True):
        """Load a rank-zero full state dict without materializing the whole model."""
        if not self._state_dict_keys:
            self._capture_state_dict_layout()

        root = not dist.is_initialized() or dist.get_rank() == 0
        if root:
            if state_dict is None:
                validation = {
                    "error": "Global rank zero requires a full state dict",
                    "missing": [],
                    "unexpected": [],
                }
            else:
                validation = self._validate_full_state_dict(state_dict, strict)
        else:
            validation = None
        if dist.is_initialized():
            payload = [validation]
            dist.broadcast_object_list(payload, src=0)
            validation = payload[0]
        if validation["error"] is not None:
            raise RuntimeError(f"Error(s) in loading state_dict: {validation['error']}")

        if self._is_checkpoint_replica():
            if not self.is_active:
                if self._full_state_root():
                    self.model.load_state_dict(state_dict, strict=strict)
            else:
                group_root = dist.get_global_rank(self.dp_shard_group, 0)
                is_group_root = self._full_state_root()
                for unit in self.units:
                    unit.all_gather()
                    try:
                        if is_group_root:
                            metas_by_name = {
                                name: meta
                                for meta in unit.param_metas
                                for name in self._parameter_state_names[id(meta.parameter)]
                            }
                            for name in self._state_dict_keys:
                                if name in metas_by_name and name in state_dict:
                                    metas_by_name[name].parameter.detach().copy_(state_dict[name])
                        for buffer in unit._full_params_buf:
                            dist.broadcast(
                                buffer,
                                src=group_root,
                                group=self.dp_shard_group,
                            )
                        unit.writeback_shards()
                    finally:
                        unit.free_full_params()
                if is_group_root:
                    self._copy_non_parameter_state(state_dict)
                for buffer in self.model.buffers():
                    self._sync_tensor(buffer, self.dp_shard_group)

        self._sync_replicated_state()
        return nn.modules.module._IncompatibleKeys(
            validation["missing"], validation["unexpected"]
        )

    def checkpoint_layout(self) -> dict[str, Any]:
        return {
            "state_dict_keys": list(self._state_dict_keys),
            "units": [
                {
                    "unit_index": unit.unit_index,
                    "parameters": [
                        {
                            "names": list(self._parameter_state_names[id(meta.parameter)]),
                            "shape": list(meta.shape),
                            "stride": list(meta.stride),
                            "numel": meta.numel,
                            "local_shape": list(meta.sharded_param._local_tensor.shape),
                            "placements": [
                                str(placement)
                                for placement in meta.sharded_param.placements
                            ],
                            "dtype": str(meta.parameter.dtype),
                            "requires_grad": meta.parameter.requires_grad,
                        }
                        for meta in unit.param_metas
                    ],
                }
                for unit in self.units
            ],
        }

    def sharded_state_dict(self) -> dict[str, Any]:
        """Return this rank's CPU model shards and replicated non-parameter state."""
        if not self._state_dict_keys:
            self._capture_state_dict_layout()
        if not self.is_active:
            return {
                "layout": self.checkpoint_layout(),
                "full_state": self.full_state_dict(),
            }
        return {
            "layout": self.checkpoint_layout(),
            "shards": [self._cpu_copy(param) for param in self.get_sharded_params()],
            "non_parameter_state": self._non_parameter_state_dict(),
        }

    def load_sharded_state_dict(self, state_dict, strict: bool = True):
        """Load same-topology local shards directly, without any collectives."""
        if not self._state_dict_keys:
            self._capture_state_dict_layout()
        if not isinstance(state_dict, dict):
            raise TypeError("Sharded checkpoint state must be a dictionary")
        if state_dict.get("layout") != self.checkpoint_layout():
            raise RuntimeError("Sharded checkpoint layout does not match the current model")
        if not self.is_active:
            return self.model.load_state_dict(state_dict["full_state"], strict=strict)

        shards = state_dict.get("shards", [])
        sharded_params = self.get_sharded_params()
        if len(shards) != len(sharded_params):
            raise RuntimeError(
                f"Expected {len(sharded_params)} parameter shards, found {len(shards)}"
            )
        for index, (parameter, shard) in enumerate(zip(sharded_params, shards)):
            if not isinstance(shard, torch.Tensor):
                raise RuntimeError(
                    f"Invalid parameter shard {index}: expected a tensor"
                )
            local = parameter._local_tensor
            if shard.shape != local.shape or shard.dtype != local.dtype:
                raise RuntimeError(
                    f"Invalid parameter shard {index}: expected shape/dtype "
                    f"{tuple(local.shape)}/{local.dtype}, got "
                    f"{tuple(shard.shape)}/{shard.dtype}"
                )

        non_parameter_state = state_dict.get("non_parameter_state", {})
        if not isinstance(non_parameter_state, dict):
            raise TypeError("Checkpoint non-parameter state must be a dictionary")
        current_state = self.model.state_dict(keep_vars=True)
        expected_names = [
            name for name in current_state if name not in self._parameter_state_keys
        ]
        actual_names = list(non_parameter_state)
        missing = [name for name in expected_names if name not in non_parameter_state]
        expected_name_set = set(expected_names)
        unexpected = [name for name in actual_names if name not in expected_name_set]
        errors = []
        if strict and missing:
            errors.append(f"Missing non-parameter key(s): {missing}")
        if strict and unexpected:
            errors.append(f"Unexpected non-parameter key(s): {unexpected}")
        for name in set(expected_names).intersection(non_parameter_state):
            destination = current_state[name]
            source = non_parameter_state[name]
            if isinstance(destination, torch.Tensor):
                if not isinstance(source, torch.Tensor):
                    errors.append(
                        f"Expected tensor for non-parameter state {name}, got "
                        f"{type(source).__name__}"
                    )
                elif source.shape != destination.shape or source.dtype != destination.dtype:
                    errors.append(
                        f"Invalid non-parameter state for {name}: expected shape/dtype "
                        f"{tuple(destination.shape)}/{destination.dtype}, got "
                        f"{tuple(source.shape)}/{source.dtype}"
                    )
        if errors:
            raise RuntimeError("; ".join(errors))

        for parameter, shard in zip(sharded_params, shards):
            parameter._local_tensor.detach().copy_(shard)
        self._copy_non_parameter_state(non_parameter_state)
        return nn.modules.module._IncompatibleKeys(missing, unexpected)

    def state_dict(self) -> Optional[OrderedDict]:
        """Compatibility alias for the memory-safe rank-zero CPU state dict."""
        return self.full_state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        """Compatibility alias for loading a rank-zero full state dict."""
        return self.load_full_state_dict(state_dict, strict=strict)

    @property
    def is_active(self) -> bool:
        return len(self.units) > 0

    def remove_hooks(self):
        for unit in self.units:
            unit.all_gather()
        for unit in self.units:
            unit.remove_hooks()

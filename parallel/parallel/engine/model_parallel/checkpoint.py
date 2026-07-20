from __future__ import annotations

import copy
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from ...state import Strategies
from .plan import ModelParallelPlan


def _cpu_copy(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", copy=True)


class ModelParallelCheckpoint:
    """Portable full and exact-topology local state for model-parallel models."""

    def __init__(self, model, pconfig, plan: ModelParallelPlan, tp_mesh, device):
        self.model = model
        self.pconfig = pconfig
        self.plan = plan
        self.tp_mesh = tp_mesh
        self.tp_group = tp_mesh.get_group()
        self.device = torch.device(device)
        parameter_ids = {id(parameter) for parameter in model.parameters()}
        self._parameter_names = {
            name
            for name, value in model.state_dict(keep_vars=True).items()
            if id(value) in parameter_ids
        }

    @property
    def is_active(self) -> bool:
        return True

    def _mesh_rank(self, dimension: Strategies) -> int:
        mesh = self.pconfig.device_mesh
        if mesh is not None and dimension in mesh.mesh_dim_names:
            return mesh.get_local_rank(dimension)
        return 0

    def _checkpoint_replica(self) -> bool:
        return self._mesh_rank(Strategies.DP_REPLICATE) == 0

    def _full_state_root(self) -> bool:
        return self._checkpoint_replica() and self.tp_mesh.get_local_rank() == 0

    def _gather_parameter(self, tensor: torch.Tensor, shard_dim: int) -> torch.Tensor:
        shards = [torch.empty_like(tensor) for _ in range(self.tp_mesh.size())]
        dist.all_gather(shards, tensor.detach(), group=self.tp_group)
        return torch.cat(shards, dim=shard_dim)

    def full_state_dict(self):
        """Collect a portable CPU state dict on global rank zero only."""
        if not self._checkpoint_replica():
            return None
        local_state = self.model.state_dict(keep_vars=True)
        root = self._full_state_root()
        result = OrderedDict() if root else None
        gathered_by_identity: dict[int, torch.Tensor] = {}

        for name, value in local_state.items():
            if name in self._parameter_names:
                placement = self.plan.placement_for_parameter(name)
                if placement.shard_dim is None:
                    full_value = value.detach() if root else None
                else:
                    full_value = self._gather_parameter(value, placement.shard_dim)
                if root:
                    identity = id(value)
                    cpu_value = gathered_by_identity.get(identity)
                    if cpu_value is None:
                        cpu_value = _cpu_copy(full_value)
                        gathered_by_identity[identity] = cpu_value
                    result[name] = cpu_value
            elif root:
                result[name] = (
                    _cpu_copy(value)
                    if isinstance(value, torch.Tensor)
                    else copy.deepcopy(value)
                )

        if root and hasattr(local_state, "_metadata"):
            result._metadata = copy.deepcopy(local_state._metadata)
        return result

    def _validate_full_state(self, state_dict, strict: bool) -> dict:
        expected_state = self.model.state_dict()
        expected = set(expected_state)
        actual = set(state_dict) if state_dict is not None else set()
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        errors = []
        if state_dict is None:
            errors.append("global rank zero requires a full state dict")
        if strict and missing:
            errors.append(f"Missing key(s): {missing}")
        if strict and unexpected:
            errors.append(f"Unexpected key(s): {unexpected}")
        if state_dict is not None:
            for name in expected.intersection(actual):
                source = state_dict[name]
                destination = expected_state[name]
                if not isinstance(source, torch.Tensor):
                    errors.append(f"Expected tensor for {name}")
                    continue
                expected_shape = list(destination.shape)
                if name in self._parameter_names:
                    placement = self.plan.placement_for_parameter(name)
                    if placement.shard_dim is not None:
                        expected_shape[placement.shard_dim] *= self.tp_mesh.size()
                if tuple(source.shape) != tuple(expected_shape):
                    errors.append(
                        f"Shape mismatch for {name}: expected {tuple(expected_shape)}, "
                        f"got {tuple(source.shape)}"
                    )
        return {
            "error": "; ".join(errors) if errors else None,
            "missing": missing,
            "unexpected": unexpected,
            "available": sorted(actual.intersection(expected)),
        }

    def _broadcast_validation(self, validation):
        if not dist.is_initialized():
            return validation
        payload = [validation if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _replica_root_tensor(self, source: torch.Tensor | None, shape, dtype):
        """Copy a full tensor from global root to TP-rank-zero of every DP replica."""
        if self.tp_mesh.get_local_rank() != 0:
            return None
        if self._checkpoint_replica():
            tensor = source.to(self.device)
        else:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
        mesh = self.pconfig.device_mesh
        if (
            mesh is not None
            and Strategies.DP_REPLICATE in mesh.mesh_dim_names
            and self.pconfig.dp_replicate_size > 1
        ):
            group = mesh.get_group(Strategies.DP_REPLICATE)
            src = dist.get_global_rank(group, 0)
            dist.broadcast(tensor, src=src, group=group)
        return tensor

    def load_full_state_dict(self, state_dict, strict: bool = True):
        root = not dist.is_initialized() or dist.get_rank() == 0
        validation = self._validate_full_state(state_dict, strict) if root else None
        validation = self._broadcast_validation(validation)
        if validation["error"]:
            raise RuntimeError(f"Error(s) in loading state_dict: {validation['error']}")

        destination_state = self.model.state_dict(keep_vars=True)
        for name, destination in destination_state.items():
            if name not in validation["available"]:
                continue

            if name in self._parameter_names:
                placement = self.plan.placement_for_parameter(name)
            else:
                placement = None
            local_shape = tuple(destination.shape)
            full_shape = list(local_shape)
            if placement is not None and placement.shard_dim is not None:
                full_shape[placement.shard_dim] *= self.tp_mesh.size()

            root_source = state_dict[name] if root and name in state_dict else None
            replica_source = self._replica_root_tensor(
                root_source, tuple(full_shape), destination.dtype
            )
            tp_root = dist.get_global_rank(self.tp_group, 0)
            if placement is not None and placement.shard_dim is not None:
                receive = torch.empty_like(destination, device=self.device)
                scatter_list = (
                    [
                        shard.contiguous()
                        for shard in replica_source.chunk(
                            self.tp_mesh.size(), dim=placement.shard_dim
                        )
                    ]
                    if self.tp_mesh.get_local_rank() == 0
                    else None
                )
                dist.scatter(
                    receive,
                    scatter_list=scatter_list,
                    src=tp_root,
                    group=self.tp_group,
                )
            else:
                receive = (
                    replica_source
                    if self.tp_mesh.get_local_rank() == 0
                    else torch.empty(
                        local_shape, dtype=destination.dtype, device=self.device
                    )
                )
                dist.broadcast(receive, src=tp_root, group=self.tp_group)
            destination.detach().copy_(receive.to(destination.device))

        return nn.modules.module._IncompatibleKeys(
            validation["missing"], validation["unexpected"]
        )

    def checkpoint_layout(self) -> dict:
        state = self.model.state_dict()
        return {
            "kind": "model-parallel",
            "plan": self.plan.checkpoint_metadata(),
            "state": [
                {
                    "name": name,
                    "local_shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "placement": (
                        self.plan.placement_for_parameter(name).parallelism.value
                        if name in self._parameter_names
                        else "buffer"
                    ),
                    "shard_dim": (
                        self.plan.placement_for_parameter(name).shard_dim
                        if name in self._parameter_names
                        else None
                    ),
                }
                for name, value in state.items()
            ],
        }

    def sharded_state_dict(self) -> dict:
        state = self.model.state_dict()
        return {
            "layout": self.checkpoint_layout(),
            "local_state": OrderedDict(
                (name, _cpu_copy(value)) for name, value in state.items()
            ),
        }

    def load_sharded_state_dict(self, state_dict, strict: bool = True):
        if not isinstance(state_dict, dict):
            raise TypeError("Sharded checkpoint state must be a dictionary")
        if state_dict.get("layout") != self.checkpoint_layout():
            raise RuntimeError("Sharded checkpoint layout does not match the current model")
        local_state = state_dict.get("local_state")
        if not isinstance(local_state, dict):
            raise TypeError("Sharded checkpoint has no local_state dictionary")
        return self.model.load_state_dict(local_state, strict=strict)


class ComposedModelParallelCheckpoint:
    """Checkpoint conversion for TP/EP semantic shards over FSDP storage shards."""

    def __init__(self, model, pconfig, plan, tp_mesh, fsdp_wrapper, device):
        self.model = model
        self.pconfig = pconfig
        self.plan = plan
        self.tp_mesh = tp_mesh
        self.tp_group = tp_mesh.get_group()
        self.fsdp = fsdp_wrapper
        self.device = torch.device(device)
        self._local_shapes = {}
        for unit in fsdp_wrapper.units:
            for meta in unit.param_metas:
                for name in fsdp_wrapper._parameter_state_names[id(meta.parameter)]:
                    self._local_shapes[name] = tuple(meta.shape)

    def _mesh_rank(self, dimension: Strategies) -> int:
        mesh = self.pconfig.device_mesh
        if mesh is not None and dimension in mesh.mesh_dim_names:
            return mesh.get_local_rank(dimension)
        return 0

    @property
    def _checkpoint_replica(self) -> bool:
        return self._mesh_rank(Strategies.DP_REPLICATE) == 0

    @property
    def _dp_shard_root(self) -> bool:
        return self._mesh_rank(Strategies.DP_SHARD) == 0

    @property
    def _portable_root(self) -> bool:
        return (
            self._checkpoint_replica
            and self._dp_shard_root
            and self.tp_mesh.get_local_rank() == 0
        )

    def _is_parameter(self, name: str) -> bool:
        return name in self.fsdp._parameter_state_keys

    def full_state_dict(self):
        local_state = self.fsdp.full_state_dict()
        if not self._checkpoint_replica or not self._dp_shard_root:
            return None
        root = self._portable_root
        result = OrderedDict() if root else None
        tp_root = dist.get_global_rank(self.tp_group, 0)

        for name in self.fsdp._state_dict_keys:
            value = local_state[name]
            if self._is_parameter(name):
                placement = self.plan.placement_for_parameter(name)
            else:
                placement = None
            if placement is not None and placement.shard_dim is not None:
                local = value.to(self.device)
                gathered = (
                    [torch.empty_like(local) for _ in range(self.tp_mesh.size())]
                    if root
                    else None
                )
                dist.gather(
                    local,
                    gather_list=gathered,
                    dst=tp_root,
                    group=self.tp_group,
                )
                if root:
                    result[name] = _cpu_copy(
                        torch.cat(gathered, dim=placement.shard_dim)
                    )
            elif root:
                result[name] = _cpu_copy(value)
        if root and hasattr(local_state, "_metadata"):
            result._metadata = copy.deepcopy(local_state._metadata)
        return result

    def _validate_portable_state(self, state_dict, strict: bool) -> dict:
        expected = set(self.fsdp._state_dict_keys)
        actual = set(state_dict) if state_dict is not None else set()
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        errors = []
        if state_dict is None:
            errors.append("global rank zero requires a portable state dict")
        if strict and missing:
            errors.append(f"Missing key(s): {missing}")
        if strict and unexpected:
            errors.append(f"Unexpected key(s): {unexpected}")
        if state_dict is not None:
            current_state = self.model.state_dict()
            for name in expected.intersection(actual):
                source = state_dict[name]
                if not isinstance(source, torch.Tensor):
                    errors.append(f"Expected tensor for {name}")
                    continue
                if self._is_parameter(name):
                    shape = list(self._local_shapes[name])
                    placement = self.plan.placement_for_parameter(name)
                    if placement.shard_dim is not None:
                        shape[placement.shard_dim] *= self.tp_mesh.size()
                else:
                    shape = list(current_state[name].shape)
                if tuple(source.shape) != tuple(shape):
                    errors.append(
                        f"Shape mismatch for {name}: expected {tuple(shape)}, "
                        f"got {tuple(source.shape)}"
                    )
        return {
            "error": "; ".join(errors) if errors else None,
            "missing": missing,
            "unexpected": unexpected,
            "available": sorted(expected.intersection(actual)),
        }

    def _broadcast_validation(self, validation):
        payload = [validation if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _local_state_from_portable(self, state_dict, validation):
        if not self._checkpoint_replica or not self._dp_shard_root:
            return None
        root = self._portable_root
        tp_root = dist.get_global_rank(self.tp_group, 0)
        local_state = OrderedDict()
        current_state = self.model.state_dict()

        for name in self.fsdp._state_dict_keys:
            if name not in validation["available"]:
                continue
            if self._is_parameter(name):
                placement = self.plan.placement_for_parameter(name)
                local_shape = self._local_shapes[name]
            else:
                placement = None
                local_shape = tuple(current_state[name].shape)
            source = state_dict[name].to(self.device) if root else None
            receive = torch.empty(
                local_shape,
                dtype=(state_dict[name].dtype if root else current_state[name].dtype),
                device=self.device,
            )
            if placement is not None and placement.shard_dim is not None:
                scatter_list = (
                    [
                        shard.contiguous()
                        for shard in source.chunk(
                            self.tp_mesh.size(), dim=placement.shard_dim
                        )
                    ]
                    if root
                    else None
                )
                dist.scatter(
                    receive,
                    scatter_list=scatter_list,
                    src=tp_root,
                    group=self.tp_group,
                )
            else:
                if root:
                    receive.copy_(source)
                dist.broadcast(receive, src=tp_root, group=self.tp_group)
            local_state[name] = _cpu_copy(receive)
        return local_state

    def load_full_state_dict(self, state_dict, strict: bool = True):
        validation = (
            self._validate_portable_state(state_dict, strict)
            if dist.get_rank() == 0
            else None
        )
        validation = self._broadcast_validation(validation)
        if validation["error"] is not None:
            raise RuntimeError(
                f"Error(s) in loading portable state_dict: {validation['error']}"
            )
        local_state = self._local_state_from_portable(state_dict, validation)
        result = self.fsdp.load_local_full_state_dict(local_state, strict=strict)
        return nn.modules.module._IncompatibleKeys(
            validation["missing"], validation["unexpected"]
        )

    def checkpoint_layout(self) -> dict:
        return {
            "kind": "model-parallel-fsdp",
            "plan": self.plan.checkpoint_metadata(),
            "fsdp": self.fsdp.checkpoint_layout(),
        }

    def sharded_state_dict(self) -> dict:
        return {
            "layout": self.checkpoint_layout(),
            "fsdp": self.fsdp.sharded_state_dict(),
        }

    def load_sharded_state_dict(self, state_dict, strict: bool = True):
        if not isinstance(state_dict, dict):
            raise TypeError("Sharded checkpoint state must be a dictionary")
        if state_dict.get("layout") != self.checkpoint_layout():
            raise RuntimeError("Composite checkpoint layout does not match the model")
        return self.fsdp.load_sharded_state_dict(state_dict["fsdp"], strict=strict)

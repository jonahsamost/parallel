from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from ..profiling import profile


@dataclass
class ParamMeta:
    """FSDP state for one logical model parameter."""

    name: str
    shape: torch.Size
    stride: tuple[int, ...]
    numel: int
    parameter: nn.Parameter
    module_refs: list[tuple[nn.Module, str]] = field(default_factory=list)
    sharded_param: Optional[nn.Parameter] = None
    local_rows: int = 0
    padded_rows: int = 0
    padded_shard: Optional[torch.Tensor] = None


class DPParamUnit:
    """One FSDP unit with per-parameter DTensor persistent state."""

    def __init__(
        self,
        module: nn.Module,
        dp_shard_mesh: DeviceMesh,
        unit_index: int,
        cpu_offload: bool = False,
        use_activation_checkpoint: bool = False,
        device: Optional[torch.device] = None,
        named_parameters: Optional[Sequence[tuple[str, nn.Parameter]]] = None,
        reshard_after_forward: bool = True,
    ):
        self.module = module
        self.mesh = dp_shard_mesh
        self.group = dp_shard_mesh.get_group()
        self.unit_index = unit_index
        self.cpu_offload = cpu_offload
        self.use_activation_checkpoint = use_activation_checkpoint
        self.reshard_after_forward = reshard_after_forward
        if named_parameters is None:
            named_parameters = list(module.named_parameters())
        self._named_params = list(named_parameters)
        if not self._named_params:
            raise ValueError("DPParamUnit requires at least one parameter")
        requires_grad = {parameter.requires_grad for _, parameter in self._named_params}
        if len(requires_grad) != 1:
            raise ValueError("A DPParamUnit cannot mix trainable and frozen parameters")
        self.requires_grad = requires_grad.pop()
        self.device = device or self._named_params[0][1].device

        self.world_size = self.mesh.size()
        self.rank = self.mesh.get_local_rank()

        self.param_metas: list[ParamMeta] = []
        self._params = [param for _, param in self._named_params]
        self.next_fwd: Optional[DPParamUnit] = None
        self.next_bwd: Optional[DPParamUnit] = None

        self._is_sharded = False
        self._is_in_backward = False
        self._full_params_buf: Optional[list[torch.Tensor]] = None
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._post_accumulate_handles: list[torch.utils.hooks.RemovableHook] = []
        self._expected_grad_param_ids: set[int] = set()
        self._post_accumulated_param_ids: set[int] = set()
        self._backward_ready_callback: Optional[Callable[[DPParamUnit], None]] = None
        self._backward_ready_notified = False
        self._gradient_reduction_started = False
        self._gradient_reduction_finished = False
        self.defer_backward_prefetch = False
        self._local_gradient_participated: Optional[bool] = None
        self._reduction_works: list[dist.Work] = []
        self._pending_grad_bufs: list[torch.Tensor] = []
        self._pending_shard_grads: list[torch.Tensor] = []
        self._original_forward = None

        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._prefetch_event: Optional[torch.cuda.Event] = None
        if self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)

    @property
    def parameter_ids(self) -> set[int]:
        return {id(parameter) for parameter in self._params}

    @property
    def backward_ready(self) -> bool:
        return self._backward_ready_notified

    def prepare_backward(self, used_parameter_ids: set[int]) -> bool:
        """Select this graph's expected gradients and report an unused unit."""
        if self._gradient_reduction_started:
            raise RuntimeError(
                f"Cannot prepare FSDP unit {self.unit_index} during gradient reduction"
            )
        self._expected_grad_param_ids = self.parameter_ids.intersection(used_parameter_ids)
        self._post_accumulated_param_ids.clear()
        self._backward_ready_notified = not self._expected_grad_param_ids
        return self._backward_ready_notified

    def missing_expected_grad_names(self) -> list[str]:
        missing_ids = self._expected_grad_param_ids - self._post_accumulated_param_ids
        return [
            meta.name for meta in self.param_metas if id(meta.parameter) in missing_ids
        ]

    # Initial sharding

    def shard(self) -> None:
        """Replace each model parameter with a dim-0 sharded DTensor parameter."""
        if self._is_sharded:
            raise RuntimeError(f"FSDP unit {self.unit_index} is already sharded")

        self.param_metas.clear()
        for name, parameter in self._named_params:
            if isinstance(parameter, DTensor):
                raise NotImplementedError(
                    "DPParamUnit does not yet compose an existing TP/EP DTensor; "
                    "apply this DP implementation to plain parameters"
                )
            if parameter.ndim == 0:
                raise ValueError(
                    f"FSDP parameter {name} must have at least one dimension"
                )
            if not parameter.is_contiguous():
                raise ValueError(f"FSDP parameter {name} must be contiguous")

            module_refs = self._find_module_refs(parameter)
            if not module_refs:
                raise RuntimeError(f"Could not find module registration for parameter {name}")
            meta = ParamMeta(
                name=name,
                shape=parameter.shape,
                stride=parameter.stride(),
                numel=parameter.numel(),
                parameter=parameter,
                module_refs=module_refs,
            )
            self._init_sharded_parameter(meta)
            self.param_metas.append(meta)

        if self.use_activation_checkpoint:
            self._apply_activation_checkpoint()
        self._register_hooks()
        self._is_sharded = True

    def _find_module_refs(self, parameter: nn.Parameter) -> list[tuple[nn.Module, str]]:
        refs: list[tuple[nn.Module, str]] = []
        for module in self.module.modules():
            for name, candidate in module._parameters.items():
                if candidate is parameter:
                    refs.append((module, name))
        return refs

    def _set_registered_parameter(self, meta: ParamMeta, parameter: nn.Parameter) -> None:
        for module, name in meta.module_refs:
            module._parameters[name] = parameter

    def _init_sharded_parameter(self, meta: ParamMeta) -> None:
        full = meta.parameter.detach().to(self.device).contiguous()
        source_rank = dist.get_global_rank(self.group, 0)
        dist.broadcast(full, src=source_rank, group=self.group)

        meta.padded_rows = (meta.shape[0] + self.world_size - 1) // self.world_size
        start = min(self.rank * meta.padded_rows, meta.shape[0])
        meta.local_rows = min(meta.padded_rows, meta.shape[0] - start)
        padded_shape = (meta.padded_rows, *meta.shape[1:])
        padded_shard = torch.zeros(padded_shape, dtype=full.dtype, device=full.device)
        if meta.local_rows:
            padded_shard[: meta.local_rows].copy_(
                full.narrow(0, start, meta.local_rows)
            )
        if self.cpu_offload:
            padded_shard = padded_shard.to("cpu").pin_memory()
        meta.padded_shard = padded_shard

        local_shard = padded_shard.narrow(0, 0, meta.local_rows)
        sharded_tensor = DTensor.from_local(
            local_shard,
            device_mesh=self.mesh,
            placements=(Shard(0),),
            run_check=False,
            shape=meta.shape,
            stride=meta.stride,
        )
        meta.sharded_param = nn.Parameter(
            sharded_tensor, requires_grad=meta.parameter.requires_grad
        )
        self._set_registered_parameter(meta, meta.sharded_param)
        self._free_unsharded_parameter(meta)

    def _free_unsharded_parameter(self, meta: ParamMeta) -> None:
        meta.parameter.data = torch.empty(
            0, dtype=meta.parameter.dtype, device=self.device
        )

    def _local_shard(self, meta: ParamMeta) -> torch.Tensor:
        if meta.sharded_param is None:
            raise RuntimeError(f"Parameter {meta.name} has not been sharded")
        # FSDP owns this storage and updates it outside autograd. ``to_local()``
        # returns an autograd view that intentionally rejects those updates.
        return meta.sharded_param._local_tensor

    def _padded_local_shard(self, meta: ParamMeta) -> torch.Tensor:
        local = self._local_shard(meta).detach()
        padded_shape = (meta.padded_rows, *meta.shape[1:])
        padded = torch.zeros(padded_shape, dtype=local.dtype, device=local.device)
        if meta.local_rows:
            padded[: meta.local_rows].copy_(local)
        if self.cpu_offload:
            padded = padded.to(self.device, non_blocking=True)
        return padded

    # Collectives

    def all_gather(self) -> None:
        """All-gather every DTensor shard and register full compute parameters."""
        if self._full_params_buf is not None:
            self.wait_for_prefetch()
            return
        if not self._is_sharded:
            raise RuntimeError("Cannot all-gather before sharding")

        full_buffers: list[torch.Tensor] = []
        with profile("all-gather"):
            for meta in self.param_metas:
                local = self._padded_local_shard(meta)
                gathered_shape = (meta.padded_rows * self.world_size, *meta.shape[1:])
                gathered = torch.empty(
                    gathered_shape, dtype=local.dtype, device=self.device
                )
                dist.all_gather_single(gathered, local, group=self.group)
                full = gathered.narrow(0, 0, meta.shape[0])
                meta.parameter.data = full
                self._set_registered_parameter(meta, meta.parameter)
                full_buffers.append(gathered)
        self._full_params_buf = full_buffers

    def wait_for_prefetch(self) -> None:
        """Block the default stream until a prefetch all-gather completes."""
        if self._prefetch_event is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_event(self._prefetch_event)
            if self._full_params_buf is not None:
                for buffer in self._full_params_buf:
                    buffer.record_stream(current_stream)
            self._prefetch_event = None

    def free_full_params(self) -> None:
        """Restore sharded DTensors to the module and release full storage."""
        if self._full_params_buf is None:
            return
        self.wait_for_prefetch()
        for meta in self.param_metas:
            if meta.sharded_param is None:
                raise RuntimeError(f"Parameter {meta.name} has no sharded state")
            self._set_registered_parameter(meta, meta.sharded_param)
            self._free_unsharded_parameter(meta)
        self._full_params_buf = None

    def start_gradient_reduction(self) -> None:
        """Launch one reduce-scatter per logical parameter asynchronously."""
        if not self.requires_grad:
            raise RuntimeError("Frozen parameter units do not have gradients to reduce")
        if self._gradient_reduction_started:
            raise RuntimeError(
                f"Gradient reduction already started for FSDP unit {self.unit_index}"
            )

        self._local_gradient_participated = any(
            meta.parameter.grad is not None for meta in self.param_metas
        )
        self._reduction_works.clear()
        self._pending_grad_bufs.clear()
        self._pending_shard_grads.clear()
        for meta in self.param_metas:
            grad_shape = (meta.padded_rows * self.world_size, *meta.shape[1:])
            grad_buf = torch.zeros(
                grad_shape, dtype=meta.parameter.dtype, device=self.device
            )
            if meta.parameter.grad is not None:
                grad_buf.narrow(0, 0, meta.shape[0]).copy_(meta.parameter.grad)
            shard_grad = torch.empty(
                (meta.padded_rows, *meta.shape[1:]),
                dtype=grad_buf.dtype,
                device=self.device,
            )
            work = dist.reduce_scatter_single(
                shard_grad,
                grad_buf,
                op=dist.ReduceOp.SUM,
                group=self.group,
                async_op=True,
            )
            meta.parameter.grad = None
            self._reduction_works.append(work)
            self._pending_grad_bufs.append(grad_buf)
            self._pending_shard_grads.append(shard_grad)

        self.free_full_params()
        self._gradient_reduction_started = True

    def finish_gradient_reduction(self) -> None:
        """Wait for reduce-scatters and release their full-sized inputs."""
        if not self._gradient_reduction_started or self._gradient_reduction_finished:
            return
        if (
            self._local_gradient_participated is None
            or len(self._reduction_works) != len(self.param_metas)
            or len(self._pending_shard_grads) != len(self.param_metas)
        ):
            raise RuntimeError(
                f"Incomplete reduction state for FSDP unit {self.unit_index}"
            )

        for work in self._reduction_works:
            work.wait()
        for index, shard_grad in enumerate(self._pending_shard_grads):
            shard_grad.div_(self.world_size)
            if self.cpu_offload:
                self._pending_shard_grads[index] = shard_grad.to("cpu")
        self._reduction_works.clear()
        self._pending_grad_bufs.clear()
        self._gradient_reduction_finished = True

    def accumulate_reduced_gradient(self, accumulate: bool) -> None:
        """Attach reduced DTensor gradients to persistent sharded parameters."""
        self.finish_gradient_reduction()
        if len(self._pending_shard_grads) != len(self.param_metas):
            raise RuntimeError(
                f"Missing reduced gradients for FSDP unit {self.unit_index}"
            )
        if accumulate:
            for meta, padded_grad in zip(self.param_metas, self._pending_shard_grads):
                if meta.sharded_param is None:
                    raise RuntimeError(f"Parameter {meta.name} has no sharded state")
                local_grad = padded_grad.narrow(0, 0, meta.local_rows)
                grad = DTensor.from_local(
                    local_grad,
                    device_mesh=self.mesh,
                    placements=(Shard(0),),
                    run_check=False,
                    shape=meta.shape,
                    stride=meta.stride,
                )
                if meta.sharded_param.grad is None:
                    meta.sharded_param.grad = grad
                else:
                    meta.sharded_param.grad._local_tensor.add_(local_grad)

        self._local_gradient_participated = None
        self._pending_shard_grads.clear()

    def reset_backward_state(self) -> None:
        """Prepare hook bookkeeping for the next backward invocation."""
        if self._gradient_reduction_started and self._reduction_works:
            raise RuntimeError(
                f"Cannot reset FSDP unit {self.unit_index} with an in-flight reduction"
            )
        self._expected_grad_param_ids.clear()
        self._post_accumulated_param_ids.clear()
        self._backward_ready_notified = False
        self._gradient_reduction_started = False
        self._gradient_reduction_finished = False

    @property
    def local_gradient_participated(self) -> bool:
        if self._local_gradient_participated is None:
            raise RuntimeError(
                f"Gradient reduction has not started for FSDP unit {self.unit_index}"
            )
        return self._local_gradient_participated

    def set_backward_ready_callback(
        self, callback: Callable[[DPParamUnit], None]
    ) -> None:
        self._backward_ready_callback = callback

    # Prefetching

    def prefetch(self) -> None:
        """
        Start an all-gather on a side stream to overlap communication.

        the all-gather reads the local parameter shard. But work already queued on 
        current stream might still be producing or modifying that shard. For instance, 
        the optimizer step might be updating it.  
        
        """
        
        if self._full_params_buf is not None:
            return
        if self._prefetch_stream is None:
            self.all_gather()
            return

        current_stream = torch.cuda.current_stream(self.device)
        self._prefetch_stream.wait_stream(current_stream)
        with torch.cuda.stream(self._prefetch_stream):
            self.all_gather()
            self._prefetch_event = torch.cuda.Event()
            self._prefetch_event.record(self._prefetch_stream)

    def _prefetch_next_forward(self) -> None:
        if self.next_fwd is not None:
            self.next_fwd.prefetch()

    def _prefetch_next_backward(self) -> None:
        if self.next_bwd is not None:
            self.next_bwd.prefetch()

    # Hook registration

    def _register_hooks(self) -> None:
        self._hook_handles.append(
            self.module.register_forward_pre_hook(self._pre_fwd_hook)
        )
        self._hook_handles.append(
            self.module.register_forward_hook(self._post_fwd_hook, always_call=True)
        )
        if self.reshard_after_forward:
            self._hook_handles.append(
                self.module.register_full_backward_pre_hook(self._pre_bwd_hook)
            )
            self._hook_handles.append(
                self.module.register_full_backward_hook(self._post_bwd_hook)
            )
        if self.requires_grad:
            for parameter in self._params:
                self._post_accumulate_handles.append(
                    parameter.register_post_accumulate_grad_hook(
                        self._post_accumulate_grad_hook
                    )
                )

    # Hooks

    def _pre_fwd_hook(self, module, args):
        if self._is_in_backward:
            return args
        self.wait_for_prefetch()
        self.all_gather()
        self._prefetch_next_forward()
        return args

    def _post_fwd_hook(self, module, args, output):
        if self._is_in_backward:
            return output
        if self.reshard_after_forward or not torch.is_grad_enabled() or output is None:
            self.free_full_params()
        return output

    def _pre_bwd_hook(self, module, grad_output):
        self._is_in_backward = True
        self.wait_for_prefetch()
        self.all_gather()
        if not self.defer_backward_prefetch:
            self._prefetch_next_backward()

    def _post_bwd_hook(self, module, grad_input, grad_output):
        if not self.requires_grad:
            self.free_full_params()
        self._is_in_backward = False

    def _post_accumulate_grad_hook(self, parameter: nn.Parameter) -> None:
        parameter_id = id(parameter)
        if parameter_id not in self._expected_grad_param_ids:
            return
        self._post_accumulated_param_ids.add(parameter_id)
        if (
            self._post_accumulated_param_ids == self._expected_grad_param_ids
            and not self._backward_ready_notified
        ):
            self._backward_ready_notified = True
            self.free_full_params()
            if self._backward_ready_callback is not None:
                self._backward_ready_callback(self)

    # Activation checkpointing

    def _apply_activation_checkpoint(self) -> None:
        self._original_forward = self.module.forward

        def _checkpointed_forward(*args, **kwargs):
            return activation_checkpoint(
                self._original_forward, *args, use_reentrant=False, **kwargs
            )

        self.module.forward = _checkpointed_forward

    # Checkpoint helpers

    def writeback_shards(self) -> None:
        """Copy currently gathered parameters into their local DTensor shards."""
        if self._full_params_buf is None:
            raise RuntimeError("Parameters must be gathered before writing back shards")
        for meta in self.param_metas:
            start = min(self.rank * meta.padded_rows, meta.shape[0])
            local = self._local_shard(meta)
            if meta.padded_shard is not None:
                meta.padded_shard.zero_()
            if meta.local_rows:
                local.copy_(meta.parameter.detach().narrow(0, start, meta.local_rows))

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for handle in self._post_accumulate_handles:
            handle.remove()
        self._post_accumulate_handles.clear()
        if self._original_forward is not None:
            self.module.forward = self._original_forward
            self._original_forward = None

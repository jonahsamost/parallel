from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from ..profiling import collective_profile


def _all_gather_contiguous(output, input, *, group):
    """Use the contiguous all-gather name available in this PyTorch release."""
    collective = getattr(dist, "all_gather_single", None)
    if collective is None:
        collective = dist.all_gather_into_tensor
    return collective(output, input, group=group)


def _reduce_scatter_contiguous(output, input, *, op, group, async_op=False):
    """Use the contiguous reduce-scatter name available in this release."""
    collective = getattr(dist, "reduce_scatter_single", None)
    if collective is None:
        collective = dist.reduce_scatter_tensor
    return collective(
        output,
        input,
        op=op,
        group=group,
        async_op=async_op,
    )


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
    bucket_offset: int = 0
    padded_numel: int = 0
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
        overlap_backward_reductions: bool = True,
    ):
        self.module = module
        self.mesh = dp_shard_mesh
        self.group = dp_shard_mesh.get_group()
        self.unit_index = unit_index
        self.cpu_offload = cpu_offload
        self.use_activation_checkpoint = use_activation_checkpoint
        self.reshard_after_forward = reshard_after_forward
        self.gradient_reduction_mode = (
            "overlapped" if overlap_backward_reductions else "deferred"
        )
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
        self._bucket_numel = 0
        self._local_param_bucket: Optional[torch.Tensor] = None
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
        self.managed_backward_unshard = False
        self._backward_unshard_scheduled = False
        self._local_gradient_participation: Optional[list[bool]] = None
        self._reduction_works: list[dist.Work] = []
        self._reduction_profiles = []
        self._pending_grad_bufs: list[torch.Tensor] = []
        self._pending_shard_grads: list[torch.Tensor] = []
        self._pending_shard_grad_bucket: Optional[torch.Tensor] = None
        self._accumulated_shard_grad_bucket_ref: Optional[
            weakref.ReferenceType[torch.Tensor]
        ] = None
        self._reused_accumulated_grad_bucket = False
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

    def prepare_backward(self, used_parameter_ids: Optional[set[int]]) -> bool:
        """Select locally expected gradients and report an unused unit.

        ``None`` is the fast-path contract: every parameter in this unit is
        expected to participate. A concrete set comes from autograd-graph
        discovery and may contain only a subset of the unit's parameters.
        """
        if self._gradient_reduction_started:
            raise RuntimeError(
                f"Cannot prepare FSDP unit {self.unit_index} during gradient reduction"
            )
        self._expected_grad_param_ids = (
            self.parameter_ids
            if used_parameter_ids is None
            else self.parameter_ids.intersection(used_parameter_ids)
        )
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
            meta.padded_rows = (meta.shape[0] + self.world_size - 1) // self.world_size
            start = min(self.rank * meta.padded_rows, meta.shape[0])
            meta.local_rows = min(meta.padded_rows, meta.shape[0] - start)
            self.param_metas.append(meta)

        self._build_bucket_layout()
        first = self.param_metas[0].parameter
        local_bucket = torch.zeros(
            self._bucket_numel,
            dtype=first.dtype,
            device=self.device,
        )
        if self.cpu_offload:
            local_bucket = local_bucket.to("cpu").pin_memory()
        self._local_param_bucket = local_bucket
        for meta in self.param_metas:
            self._init_sharded_parameter(meta)
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
        if self._local_param_bucket is None:
            raise RuntimeError("FSDP parameter bucket has not been allocated")
        full = meta.parameter.detach().to(self.device).contiguous()
        source_rank = dist.get_global_rank(self.group, 0)
        dist.broadcast(full, src=source_rank, group=self.group)

        start = min(self.rank * meta.padded_rows, meta.shape[0])
        padded_shape = (meta.padded_rows, *meta.shape[1:])
        padded_shard = self._local_param_bucket.narrow(
            0, meta.bucket_offset, meta.padded_numel
        ).view(padded_shape)
        if meta.local_rows:
            source = full.narrow(0, start, meta.local_rows)
            if source.device != padded_shard.device:
                source = source.to(padded_shard.device)
            padded_shard[: meta.local_rows].copy_(source)
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

    def _build_bucket_layout(self) -> None:
        """Assign every padded local parameter shard a flat unit offset."""
        offset = 0
        for meta in self.param_metas:
            meta.bucket_offset = offset
            meta.padded_numel = meta.padded_rows
            for size in meta.shape[1:]:
                meta.padded_numel *= size
            offset += meta.padded_numel
        self._bucket_numel = offset

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

    def _packed_local_shards(self) -> torch.Tensor:
        """Return the flat local parameter bucket on the collective device."""
        if self._local_param_bucket is None:
            raise RuntimeError("FSDP parameter bucket has not been allocated")
        packed = self._local_param_bucket.detach()
        if packed.device != self.device:
            packed = packed.to(self.device, non_blocking=True)
        return packed

    def _unpack_full_parameters(
        self, gathered: torch.Tensor
    ) -> list[torch.Tensor]:
        """Turn rank-major gathered shards into contiguous full parameters."""
        rank_buckets = gathered.view(self.world_size, self._bucket_numel)
        full_buffers = []
        for meta in self.param_metas:
            rank_shards = rank_buckets.narrow(
                1, meta.bucket_offset, meta.padded_numel
            )
            # A column slice of the rank-major bucket is not contiguous when
            # the unit has multiple parameters. Materialize one contiguous
            # compute buffer so Linear/grouped-GEMM weights retain their
            # original layout.
            full_padded = rank_shards.contiguous().view(
                meta.padded_rows * self.world_size, *meta.shape[1:]
            )
            full = full_padded.narrow(0, 0, meta.shape[0])
            meta.parameter.data = full
            self._set_registered_parameter(meta, meta.parameter)
            full_buffers.append(full_padded)
        return full_buffers

    # Collectives

    def all_gather(self, *, mode: str = "demand") -> None:
        """All-gather every DTensor shard and register full compute parameters."""
        if self._full_params_buf is not None:
            self.wait_for_prefetch()
            return
        if not self._is_sharded:
            raise RuntimeError("Cannot all-gather before sharding")

        local_bucket = self._packed_local_shards()
        gathered = torch.empty(
            self._bucket_numel * self.world_size,
            dtype=local_bucket.dtype,
            device=self.device,
        )
        first_name = self.param_metas[0].name
        detail = (
            f"unit{self.unit_index}:{first_name}"
            f"+{len(self.param_metas) - 1}params"
        )
        with collective_profile(
            "fsdp_all_gather",
            value=local_bucket,
            mode=mode,
            detail=detail,
        ):
            _all_gather_contiguous(gathered, local_bucket, group=self.group)
        self._full_params_buf = self._unpack_full_parameters(gathered)

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
        """Launch one packed reduce-scatter for the entire unit asynchronously."""
        if not self.requires_grad:
            raise RuntimeError("Frozen parameter units do not have gradients to reduce")
        if self._gradient_reduction_started:
            raise RuntimeError(
                f"Gradient reduction already started for FSDP unit {self.unit_index}"
            )

        self._local_gradient_participation = [
            meta.parameter.grad is not None for meta in self.param_metas
        ]
        self._reduction_works.clear()
        self._reduction_profiles.clear()
        self._pending_grad_bufs.clear()
        self._pending_shard_grads.clear()
        self._pending_shard_grad_bucket = None
        self._reused_accumulated_grad_bucket = False

        # reduce_scatter_tensor splits its input into one contiguous chunk per
        # destination rank. Pack every parameter's destination shard into that
        # rank-major layout so the entire unit reduces in one collective.
        first = self.param_metas[0].parameter
        grad_bucket = torch.zeros(
            self.world_size,
            self._bucket_numel,
            dtype=first.dtype,
            device=self.device,
        )
        for meta in self.param_metas:
            if meta.parameter.grad is None:
                continue
            gradient = meta.parameter.grad
            for destination_rank in range(self.world_size):
                source_start = destination_rank * meta.padded_rows
                rows = min(
                    meta.padded_rows,
                    max(meta.shape[0] - source_start, 0),
                )
                if not rows:
                    continue
                destination = grad_bucket[destination_rank].narrow(
                    0, meta.bucket_offset, meta.padded_numel
                ).view(meta.padded_rows, *meta.shape[1:])
                destination[:rows].copy_(
                    gradient.narrow(0, source_start, rows)
                )

        accumulated_bucket = (
            self._accumulated_shard_grad_bucket_ref()
            if self._accumulated_shard_grad_bucket_ref is not None
            else None
        )
        has_accumulated_gradients = any(
            meta.sharded_param is not None and meta.sharded_param.grad is not None
            for meta in self.param_metas
        )
        if (
            accumulated_bucket is not None
            and accumulated_bucket.device == self.device
            and has_accumulated_gradients
        ):
            # Each rank owns exactly one output shard from the previous
            # microbatch. Add that shard, scaled by the later averaging
            # divisor, only to this rank's destination chunk. The collective
            # can then overwrite the same persistent gradient bucket with
            # ``old_average + new_average`` instead of allocating a second
            # local-model-sized output bucket.
            local_destination = grad_bucket[self.rank]
            for meta in self.param_metas:
                if meta.sharded_param is None or meta.sharded_param.grad is None:
                    continue
                destination = local_destination.narrow(
                    0, meta.bucket_offset, meta.padded_numel
                ).view(meta.padded_rows, *meta.shape[1:])
                destination[: meta.local_rows].add_(
                    meta.sharded_param.grad._local_tensor,
                    alpha=self.world_size,
                )
            shard_grad_bucket = accumulated_bucket
            self._reused_accumulated_grad_bucket = True
        else:
            shard_grad_bucket = torch.empty(
                self._bucket_numel,
                dtype=grad_bucket.dtype,
                device=self.device,
            )
            if not has_accumulated_gradients:
                self._accumulated_shard_grad_bucket_ref = None
        first_name = self.param_metas[0].name
        detail = (
            f"unit{self.unit_index}:{first_name}"
            f"+{len(self.param_metas) - 1}params"
        )
        reduction_profile = collective_profile(
            "fsdp_reduce_scatter",
            value=grad_bucket,
            mode=self.gradient_reduction_mode,
            detail=detail,
            defer_cuda_end=True,
        )
        with reduction_profile:
            work = _reduce_scatter_contiguous(
                shard_grad_bucket,
                grad_bucket.view(-1),
                op=dist.ReduceOp.SUM,
                group=self.group,
                async_op=True,
            )

        for meta in self.param_metas:
            meta.parameter.grad = None
            self._pending_shard_grads.append(
                shard_grad_bucket.narrow(
                    0, meta.bucket_offset, meta.padded_numel
                ).view(meta.padded_rows, *meta.shape[1:])
            )
        self._reduction_works.append(work)
        self._reduction_profiles.append(reduction_profile)
        self._pending_grad_bufs.append(grad_bucket)
        self._pending_shard_grad_bucket = shard_grad_bucket

        self.free_full_params()
        self._gradient_reduction_started = True

    def finish_gradient_reduction(self) -> None:
        """Wait for reduce-scatters and release their full-sized inputs."""
        if not self._gradient_reduction_started or self._gradient_reduction_finished:
            return
        if (
            self._local_gradient_participation is None
            or len(self._reduction_works) != 1
            or len(self._reduction_profiles) != 1
            or len(self._pending_shard_grads) != len(self.param_metas)
        ):
            raise RuntimeError(
                f"Incomplete reduction state for FSDP unit {self.unit_index}"
            )

        for work, reduction_profile in zip(
            self._reduction_works, self._reduction_profiles, strict=True
        ):
            work.wait()
            reduction_profile.complete()
        # Every per-parameter shard is a view into the one reduced bucket.
        if self._pending_shard_grad_bucket is None:
            raise RuntimeError(
                f"FSDP unit {self.unit_index} has no reduced gradient bucket"
            )
        self._pending_shard_grad_bucket.div_(self.world_size)
        if self.cpu_offload:
            self._pending_shard_grad_bucket = self._pending_shard_grad_bucket.to("cpu")
            self._pending_shard_grads = [
                self._pending_shard_grad_bucket.narrow(
                    0, meta.bucket_offset, meta.padded_numel
                ).view(meta.padded_rows, *meta.shape[1:])
                for meta in self.param_metas
            ]
        self._reduction_works.clear()
        self._reduction_profiles.clear()
        self._pending_grad_bufs.clear()
        self._gradient_reduction_finished = True

    def accumulate_reduced_gradient(self, accumulate: bool) -> None:
        """Attach reduced DTensor gradients to persistent sharded parameters."""
        self.accumulate_reduced_gradients([accumulate] * len(self.param_metas))

    def accumulate_reduced_gradients(self, accumulate: Sequence[bool]) -> None:
        """Attach selected reduced gradients to persistent DTensor parameters."""
        self.finish_gradient_reduction()
        if len(self._pending_shard_grads) != len(self.param_metas):
            raise RuntimeError(
                f"Missing reduced gradients for FSDP unit {self.unit_index}"
            )
        if len(accumulate) != len(self.param_metas):
            raise RuntimeError(
                f"Expected {len(self.param_metas)} participation values for FSDP "
                f"unit {self.unit_index}, got {len(accumulate)}"
            )
        for should_accumulate, meta, padded_grad in zip(
            accumulate, self.param_metas, self._pending_shard_grads
        ):
            if should_accumulate:
                if meta.sharded_param is None:
                    raise RuntimeError(f"Parameter {meta.name} has no sharded state")
                local_grad = padded_grad.narrow(0, 0, meta.local_rows)
                if meta.sharded_param.grad is None:
                    meta.sharded_param.grad = DTensor.from_local(
                        local_grad,
                        device_mesh=self.mesh,
                        placements=(Shard(0),),
                        run_check=False,
                        shape=meta.shape,
                        stride=meta.stride,
                    )
                elif not self._reused_accumulated_grad_bucket:
                    meta.sharded_param.grad._local_tensor.add_(local_grad)

        if self._pending_shard_grad_bucket is not None:
            self._accumulated_shard_grad_bucket_ref = weakref.ref(
                self._pending_shard_grad_bucket
            )
        self._local_gradient_participation = None
        self._pending_shard_grads.clear()
        self._pending_shard_grad_bucket = None
        self._reused_accumulated_grad_bucket = False

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
        self._reduction_profiles.clear()
        self.managed_backward_unshard = False
        self._backward_unshard_scheduled = False

    @property
    def local_gradient_participated(self) -> bool:
        return any(self.local_gradient_participation)

    @property
    def local_gradient_participation(self) -> list[bool]:
        if self._local_gradient_participation is None:
            raise RuntimeError(
                f"Gradient reduction has not started for FSDP unit {self.unit_index}"
            )
        return self._local_gradient_participation

    def set_backward_ready_callback(
        self, callback: Callable[[DPParamUnit], None]
    ) -> None:
        self._backward_ready_callback = callback

    # Prefetching

    def schedule_backward_unshard(self) -> None:
        """Collectively unshard this unit at its wrapper-assigned position."""
        if self._backward_unshard_scheduled:
            return
        self._backward_unshard_scheduled = True
        self.prefetch()

    def wait_for_backward_unshard(self) -> None:
        """Wait until the wrapper-scheduled full parameters are consumable."""
        if not self._backward_unshard_scheduled:
            raise RuntimeError(
                f"FSDP unit {self.unit_index} reached backward before its globally "
                "ordered parameter all-gather was scheduled"
            )
        self.wait_for_prefetch()
        if self._full_params_buf is None:
            raise RuntimeError(
                f"FSDP unit {self.unit_index} has no full parameters for backward"
            )

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
            self.all_gather(mode="prefetch")
            return

        current_stream = torch.cuda.current_stream(self.device)
        self._prefetch_stream.wait_stream(current_stream)
        with torch.cuda.stream(self._prefetch_stream):
            self.all_gather(mode="prefetch")
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
        if self.managed_backward_unshard:
            self.wait_for_backward_unshard()
        else:
            self.wait_for_prefetch()
            self.all_gather()
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

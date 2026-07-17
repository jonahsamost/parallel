from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from ..profiling import profile


@dataclass
class ParamMeta:
    """Metadata for one original parameter within the flat buffer."""

    name: str
    shape: torch.Size
    numel: int
    offset: int
    parameter: nn.Parameter


# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

class DPParamUnit:
    """ One FSDP unit — wraps a single nn.Module (typically one transformer layer). """

    def __init__(
        self,
        module: nn.Module,
        dp_shard_group: dist.ProcessGroup,
        unit_index: int,
        cpu_offload: bool = False,
        use_activation_checkpoint: bool = False,
        device: Optional[torch.device] = None,
        named_parameters: Optional[Sequence[tuple[str, nn.Parameter]]] = None,
        reshard_after_forward: bool = True,
    ):
        self.module = module
        self.group = dp_shard_group
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

        self.world_size = dist.get_world_size(group=dp_shard_group)
        self.rank = dist.get_rank(group=dp_shard_group)

        self.param_metas: list[ParamMeta] = []
        self._params = [param for _, param in self._named_params]
        self.flat_shard: Optional[nn.Parameter] = None
        self.flat_numel: int = 0
        self.padded_numel: int = 0
        self.chunk_size: int = 0

        self.next_fwd: Optional[DPParamUnit] = None
        self.next_bwd: Optional[DPParamUnit] = None

        self._is_sharded: bool = False
        self._is_in_backward: bool = False
        self._full_params_buf: Optional[torch.Tensor] = None
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._post_accumulate_handles: list[torch.utils.hooks.RemovableHook] = []
        self._expected_grad_param_ids: Optional[set[int]] = None
        self._post_accumulated_param_ids: set[int] = set()
        self._backward_ready_callback: Optional[Callable[[DPParamUnit], None]] = None
        self._backward_ready_notified: bool = False
        self._gradient_reduction_started: bool = False
        self._gradient_reduction_finished: bool = False
        self._local_gradient_participated: Optional[bool] = None
        self._reduction_work = None
        self._pending_grad_buf: Optional[torch.Tensor] = None
        self._pending_shard_grad: Optional[torch.Tensor] = None
        self._original_forward = None

        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._prefetch_event: Optional[torch.cuda.Event] = None
        if self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)

    # Initial sharding

    def shard(self):
        """Flatten all params -> pad -> keep this rank's 1/N chunk.
        Registers forward/backward hooks on self.module."""
        if self._is_sharded:
            raise RuntimeError(f"FSDP unit {self.unit_index} is already sharded")

        flat_buf = self._build_flat_buffer().to(self.device)
        source_rank = dist.get_global_rank(self.group, 0)
        dist.broadcast(flat_buf, src=source_rank, group=self.group)

        # make divisible by world size 
        self.padded_numel = (
            (self.flat_numel + self.world_size - 1)
            // self.world_size
            * self.world_size
        )
        self.chunk_size = self.padded_numel // self.world_size

        padded = torch.zeros(self.padded_numel, dtype=flat_buf.dtype, device=flat_buf.device)
        padded[: self.flat_numel] = flat_buf

        shard = padded[
            self.rank * self.chunk_size : (self.rank + 1) * self.chunk_size
        ].clone()

        if self.cpu_offload:
            shard = shard.to("cpu").pin_memory()
        else:
            shard = shard.to(self.device)
        self.flat_shard = nn.Parameter(shard, requires_grad=self.requires_grad)

        self._replace_params_with_placeholders()
        if self.use_activation_checkpoint:
            self._apply_activation_checkpoint()
        self._register_hooks()
        self._is_sharded = True

    def _build_flat_buffer(self) -> torch.Tensor:
        """Concatenate all params into a single 1D tensor, recording metadata."""
        devices = {param.device for param in self._params}
        dtypes = {param.dtype for param in self._params}
        if len(devices) != 1:
            raise ValueError(f"FSDP unit {self.unit_index} has parameters on multiple devices: {devices}")
        if len(dtypes) != 1:
            raise ValueError(f"FSDP unit {self.unit_index} has parameters with multiple dtypes: {dtypes}")

        offset = 0
        parts: list[torch.Tensor] = []
        self.param_metas.clear()
        for name, p in self._named_params:
            meta = ParamMeta(
                name=name,
                shape=p.shape,
                numel=p.numel(),
                offset=offset,
                parameter=p,
            )
            self.param_metas.append(meta)
            parts.append(p.detach().flatten())
            offset += p.numel()
        self.flat_numel = offset
        return torch.cat(parts)

    def _replace_params_with_placeholders(self):
        """Replace each param's data with a zero-storage tensor so GPU memory is freed.
        After sharding, full-sized params are not needed. If we don't replace the param data,
        both the full original params and the shard sit in GPU memory 
        """
        for meta in self.param_metas:
            meta.parameter.data = torch.empty(0, dtype=meta.parameter.dtype, device=self.device)

    # collectives

    def all_gather(self):
        """All-gather flat_shard -> full flat buffer -> view back into param shapes."""
        if self._full_params_buf is not None:
            self.wait_for_prefetch()
            return
        if self.flat_shard is None:
            raise RuntimeError("Cannot all-gather before sharding")

        with profile("all-gather"):
            shard_on_device = self.flat_shard.detach()
            if self.cpu_offload:
                shard_on_device = shard_on_device.to(self.device, non_blocking=True)

            full_buf = torch.empty(
                self.padded_numel, dtype=shard_on_device.dtype, device=self.device
            )
            dist.all_gather_into_tensor(full_buf, shard_on_device, group=self.group)

        for meta in self.param_metas:
            meta.parameter.data = full_buf[meta.offset : meta.offset + meta.numel].view(meta.shape)

        self._full_params_buf = full_buf

    def wait_for_prefetch(self):
        """Block the default stream until a prefetch all-gather completes."""
        if self._prefetch_event is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_event(self._prefetch_event)
            if self._full_params_buf is not None:
                self._full_params_buf.record_stream(current_stream)
            self._prefetch_event = None

    def free_full_params(self):
        """Discard full params, restore placeholders. GPU memory freed."""
        if self._full_params_buf is None:
            return
        self.wait_for_prefetch()
        self._replace_params_with_placeholders()
        self._full_params_buf = None

    def start_gradient_reduction(self):
        """Launch this unit's reduce-scatter asynchronously."""
        if not self.requires_grad:
            raise RuntimeError("Frozen parameter units do not have gradients to reduce")
        if self.flat_shard is None:
            raise RuntimeError("Cannot reduce-scatter before sharding")
        if self._gradient_reduction_started:
            raise RuntimeError(
                f"Gradient reduction already started for FSDP unit {self.unit_index}"
            )
        if self._expected_grad_param_ids is None:
            raise RuntimeError(
                f"FSDP unit {self.unit_index} was not prepared for backward"
            )

        local_participated = bool(self._expected_grad_param_ids)
        grad_buf = torch.zeros(
            self.padded_numel, dtype=self.flat_shard.dtype, device=self.device
        )
        for meta in self.param_metas:
            if meta.parameter.grad is not None:
                grad_buf[meta.offset : meta.offset + meta.numel] = meta.parameter.grad.flatten()

        shard_grad = torch.empty(
            self.chunk_size, dtype=grad_buf.dtype, device=self.device
        )
        reduction_work = dist.reduce_scatter_tensor(
            shard_grad,
            grad_buf,
            op=dist.ReduceOp.SUM,
            group=self.group,
            async_op=True,
        )

        for meta in self.param_metas:
            meta.parameter.grad = None
        self.free_full_params()

        # Keep the collective input and output alive until the asynchronous work
        # has completed. The wrapper waits for it before returning from
        # finalize_backward(), so the optimizer never observes an in-flight grad.
        self._local_gradient_participated = local_participated
        self._reduction_work = reduction_work
        self._pending_grad_buf = grad_buf
        self._pending_shard_grad = shard_grad
        self._gradient_reduction_started = True

    def finish_gradient_reduction(self):
        """Wait for reduce-scatter and release its full-sized input buffer."""
        if not self._gradient_reduction_started:
            return
        if self._gradient_reduction_finished:
            return
        if (
            self._local_gradient_participated is None
            or self._reduction_work is None
            or self._pending_shard_grad is None
        ):
            raise RuntimeError(
                f"Incomplete reduction state for FSDP unit {self.unit_index}"
            )

        self._reduction_work.wait()
        self._pending_shard_grad.div_(self.world_size)
        if self.cpu_offload:
            self._pending_shard_grad = self._pending_shard_grad.to("cpu")
        self._reduction_work = None
        self._pending_grad_buf = None
        self._gradient_reduction_finished = True

    def accumulate_reduced_gradient(self, accumulate: bool):
        """Accumulate the completed local shard once global participation is known."""
        self.finish_gradient_reduction()
        if self._pending_shard_grad is None:
            raise RuntimeError(
                f"Missing reduced gradient for FSDP unit {self.unit_index}"
            )
        if accumulate:
            if self.flat_shard.grad is None:
                self.flat_shard.grad = self._pending_shard_grad
            else:
                self.flat_shard.grad.add_(self._pending_shard_grad)

        self._local_gradient_participated = None
        self._pending_shard_grad = None

    def reset_backward_state(self):
        """Prepare hook bookkeeping for the next backward invocation."""
        if self._gradient_reduction_started and self._reduction_work is not None:
            raise RuntimeError(
                f"Cannot reset FSDP unit {self.unit_index} with an in-flight reduction"
            )
        self._expected_grad_param_ids = None
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

    def prepare_backward(self, used_param_ids: set[int]) -> bool:
        """Record which local parameters autograd will visit in this backward."""
        if self._expected_grad_param_ids is not None:
            raise RuntimeError(
                f"FSDP unit {self.unit_index} is already prepared for backward"
            )
        if self._gradient_reduction_started:
            raise RuntimeError(
                f"FSDP unit {self.unit_index} still has reduction state from backward"
            )
        self._expected_grad_param_ids = {
            id(parameter)
            for parameter in self._params
            if id(parameter) in used_param_ids
        }
        if not self._expected_grad_param_ids:
            self.free_full_params()
            self._backward_ready_notified = True
            return True
        return False

    @property
    def backward_ready(self) -> bool:
        return self._backward_ready_notified

    @property
    def parameter_ids(self) -> set[int]:
        return {id(parameter) for parameter in self._params}

    def missing_expected_grad_names(self) -> list[str]:
        if self._expected_grad_param_ids is None:
            return [name for name, _ in self._named_params]
        return [
            name
            for name, parameter in self._named_params
            if id(parameter) in self._expected_grad_param_ids
            and id(parameter) not in self._post_accumulated_param_ids
        ]

    # Prefetching

    def prefetch(self):
        """Start all-gather for this unit on a side stream to overlap compute"""
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

    def _prefetch_next_forward(self):
        if self.next_fwd is not None:
            self.next_fwd.prefetch()

    def _prefetch_next_backward(self):
        if self.next_bwd is not None:
            self.next_bwd.prefetch()

    # Hook registration

    def _register_hooks(self):
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
                    parameter.register_post_accumulate_grad_hook(self._post_accumulate_grad_hook)
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
        self._prefetch_next_backward()

    def _post_bwd_hook(self, module, grad_input, grad_output):
        if not self.requires_grad:
            self.free_full_params()
        self._is_in_backward = False

    def _post_accumulate_grad_hook(self, parameter):
        if self._expected_grad_param_ids is None:
            raise RuntimeError(
                "FSDP backward started without calling prepare_backward(loss)"
            )
        self._post_accumulated_param_ids.add(id(parameter))
        if (
            self._expected_grad_param_ids <= self._post_accumulated_param_ids
            and not self._backward_ready_notified
        ):
            self._backward_ready_notified = True
            self.free_full_params()
            if self._backward_ready_callback is not None:
                self._backward_ready_callback(self)

    # Activation checkpointing

    def _apply_activation_checkpoint(self):
        """Wrap the module's forward with checkpoint once at init time.

        The _is_in_backward guard prevents the forward hooks from firing
        during the recomputation pass triggered by checkpoint.
        """
        self._original_forward = self.module.forward

        def _checkpointed_forward(*args, **kwargs):
            return activation_checkpoint(
                self._original_forward, *args, use_reentrant=False, **kwargs
            )

        self.module.forward = _checkpointed_forward

    def writeback_shard(self):
        """Copy the currently gathered parameters into this rank's local shard."""
        if self._full_params_buf is None or self.flat_shard is None:
            raise RuntimeError("Parameters must be gathered before writing back a shard")
        padded = torch.zeros(
            self.padded_numel,
            dtype=self._full_params_buf.dtype,
            device=self._full_params_buf.device,
        )
        padded[: self.flat_numel].copy_(self._full_params_buf[: self.flat_numel])
        shard = padded[self.rank * self.chunk_size : (self.rank + 1) * self.chunk_size]
        if self.cpu_offload:
            shard = shard.to("cpu")
        self.flat_shard.data.copy_(shard)

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for handle in self._post_accumulate_handles:
            handle.remove()
        self._post_accumulate_handles.clear()
        if self._original_forward is not None:
            self.module.forward = self._original_forward
            self._original_forward = None

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


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
        self._post_accumulated_param_ids: set[int] = set()
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

    def reduce_scatter_grads(self, accumulate: bool = True):
        """Flatten all grads -> pad -> reduce-scatter -> each rank keeps its shard grad."""
        if not self.requires_grad:
            raise RuntimeError("Frozen parameter units do not have gradients to reduce")
        if self.flat_shard is None:
            raise RuntimeError("Cannot reduce-scatter before sharding")
        grad_buf = torch.zeros(
            self.padded_numel, dtype=self.flat_shard.dtype, device=self.device
        )
        for meta in self.param_metas:
            if meta.parameter.grad is not None:
                grad_buf[meta.offset : meta.offset + meta.numel] = meta.parameter.grad.flatten()

        shard_grad = torch.empty(
            self.chunk_size, dtype=grad_buf.dtype, device=self.device
        )
        dist.reduce_scatter_tensor(
            shard_grad, grad_buf, op=dist.ReduceOp.SUM, group=self.group
        )
        shard_grad.div_(self.world_size)

        if self.cpu_offload:
            shard_grad = shard_grad.to("cpu")

        if accumulate:
            if self.flat_shard.grad is None:
                self.flat_shard.grad = shard_grad
            else:
                self.flat_shard.grad.add_(shard_grad)

        for meta in self.param_metas:
            meta.parameter.grad = None

    def finalize_backward(self):
        """Reduce gradients after autograd has finalized every parameter gradient."""
        if not self.requires_grad:
            self.free_full_params()
            return
        local_participated = any(meta.parameter.grad is not None for meta in self.param_metas)
        participated = torch.tensor(
            int(local_participated),
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(participated, op=dist.ReduceOp.MAX, group=self.group)
        self.reduce_scatter_grads(accumulate=bool(participated.item()))
        self.free_full_params()

    # Prefetching

    def prefetch(self):
        """Start all-gather for this unit on a side stream to overlap compute"""
        if self._full_params_buf is not None:
            return
        if self._prefetch_stream is None:
            self.all_gather()
            return

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
        self._post_accumulated_param_ids.clear()
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
        self._post_accumulated_param_ids.add(id(parameter))
        if len(self._post_accumulated_param_ids) == len(self._params):
            self.free_full_params()

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

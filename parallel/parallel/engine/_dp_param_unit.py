from __future__ import annotations

from dataclasses import dataclass
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
    dtype: torch.dtype


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
    ):
        self.module = module
        self.group = dp_shard_group
        self.unit_index = unit_index
        self.cpu_offload = cpu_offload
        self.use_activation_checkpoint = use_activation_checkpoint
        self.device = device or next(module.parameters()).device

        self.world_size = dist.get_world_size(group=dp_shard_group)
        self.rank = dist.get_rank(group=dp_shard_group)

        self.param_metas: list[ParamMeta] = []
        self._params: list[nn.Parameter] = []
        self.flat_shard: Optional[torch.Tensor] = None
        self.flat_numel: int = 0
        self.padded_numel: int = 0
        self.chunk_size: int = 0

        self.next_fwd: Optional[DPParamUnit] = None
        self.next_bwd: Optional[DPParamUnit] = None

        self._is_sharded: bool = False
        self._is_in_backward: bool = False
        self._full_params_buf: Optional[torch.Tensor] = None
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []

        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._prefetch_event: Optional[torch.cuda.Event] = None
        if self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)

    # ─── Initial sharding ───────────────────────────────────────

    def shard(self):
        """Flatten all params -> pad -> keep this rank's 1/N chunk.
        Registers forward/backward hooks on self.module."""
        self._params = list(self.module.parameters())
        if not self._params:
            return

        flat_buf = self._build_flat_buffer()

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
            self.flat_shard = shard.to("cpu").pin_memory() # make it non-pageable
        else:
            self.flat_shard = shard

        self._replace_params_with_placeholders()
        if self.use_activation_checkpoint:
            self._apply_activation_checkpoint()
        self._register_hooks()
        self._is_sharded = True

    def _build_flat_buffer(self) -> torch.Tensor:
        """Concatenate all params into a single 1D tensor, recording metadata."""
        offset = 0
        parts: list[torch.Tensor] = []
        for name, p in self.module.named_parameters():
            meta = ParamMeta(
                name=name,
                shape=p.shape,
                numel=p.numel(),
                offset=offset,
                dtype=p.dtype,
            )
            self.param_metas.append(meta)
            parts.append(p.data.flatten())
            offset += p.numel()
        self.flat_numel = offset
        return torch.cat(parts)

    def _replace_params_with_placeholders(self):
        """Replace each param's data with a zero-storage tensor so GPU memory is freed.
        After sharding, full-sized params are not needed. If we don't replace the param data,
        both the full original params and the shard sit in GPU memory 
        """
        for p in self._params:
            p.data = torch.empty(0, dtype=p.dtype, device=self.device)

    # collectives

    def all_gather(self):
        """All-gather flat_shard -> full flat buffer -> view back into param shapes."""
        if self._full_params_buf is not None:
            return

        shard_on_device = self.flat_shard
        if self.cpu_offload:
            shard_on_device = self.flat_shard.to(self.device, non_blocking=True)

        full_buf = torch.empty(
            self.padded_numel, dtype=shard_on_device.dtype, device=self.device
        )
        dist.all_gather_into_tensor(full_buf, shard_on_device, group=self.group)

        for p, meta in zip(self._params, self.param_metas):
            p.data = full_buf[meta.offset : meta.offset + meta.numel].view(meta.shape)

        self._full_params_buf = full_buf

    def wait_for_prefetch(self):
        """Block the default stream until a prefetch all-gather completes."""
        if self._prefetch_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._prefetch_event)
            self._prefetch_event = None

    def free_full_params(self):
        """Discard full params, restore placeholders. GPU memory freed."""
        if self._full_params_buf is None:
            return
        self._replace_params_with_placeholders()
        self._full_params_buf = None

    def reduce_scatter_grads(self):
        """Flatten all grads -> pad -> reduce-scatter -> each rank keeps its shard grad."""
        grad_buf = torch.zeros(
            self.padded_numel, dtype=self.flat_shard.dtype, device=self.device
        )
        for p, meta in zip(self._params, self.param_metas):
            if p.grad is not None:
                grad_buf[meta.offset : meta.offset + meta.numel] = p.grad.flatten()

        shard_grad = torch.empty(
            self.chunk_size, dtype=grad_buf.dtype, device=self.device
        )
        dist.reduce_scatter_tensor(
            shard_grad, grad_buf, op=dist.ReduceOp.AVG, group=self.group
        )

        if self.cpu_offload:
            self.flat_shard.grad = shard_grad.to("cpu", non_blocking=True)
        else:
            self.flat_shard.grad = shard_grad

        for p in self._params:
            p.grad = None

    # CPU offloading

    def _shard_to_device(self) -> torch.Tensor:
        """Copy flat_shard from pinned CPU memory to GPU (non-blocking)."""
        return self.flat_shard.to(self.device, non_blocking=True)

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
            self.module.register_forward_hook(self._post_fwd_hook)
        )
        self._hook_handles.append(
            self.module.register_full_backward_pre_hook(self._pre_bwd_hook)
        )
        self._hook_handles.append(
            self.module.register_full_backward_hook(self._post_bwd_hook)
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
        self.free_full_params()
        return output

    def _pre_bwd_hook(self, module, grad_output):
        self._is_in_backward = True
        self.wait_for_prefetch()
        self.all_gather()
        self._prefetch_next_backward()

    def _post_bwd_hook(self, module, grad_input, grad_output):
        self.free_full_params()
        self.reduce_scatter_grads()
        self._is_in_backward = False

    # Activation checkpointing

    def _apply_activation_checkpoint(self):
        """Wrap the module's forward with checkpoint once at init time.

        The _is_in_backward guard prevents the forward hooks from firing
        during the recomputation pass triggered by checkpoint.
        """
        original_forward = self.module.forward

        def _checkpointed_forward(*args, **kwargs):
            return activation_checkpoint(
                original_forward, *args, use_reentrant=False, **kwargs
            )

        self.module.forward = _checkpointed_forward

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

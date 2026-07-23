from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ...profiling import collective_profile
from .token_dispatch import TokenDispatcher


@dataclass(frozen=True)
class ExpertPartition:
    """Contiguous expert ownership and replicated-token route localization."""

    num_experts: int
    ep_size: int
    ep_rank: int

    def __post_init__(self):
        if self.ep_size < 1:
            raise ValueError("ep_size must be positive")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("ep_rank is outside the expert-parallel group")
        if self.num_experts % self.ep_size:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"ep_size ({self.ep_size})"
            )

    @property
    def local_count(self) -> int:
        return self.num_experts // self.ep_size

    @property
    def start(self) -> int:
        return self.ep_rank * self.local_count

    @property
    def stop(self) -> int:
        return self.start + self.local_count

    @property
    def global_experts(self) -> range:
        return range(self.start, self.stop)

    def owner(self, global_expert: torch.Tensor) -> torch.Tensor:
        return torch.div(global_expert, self.local_count, rounding_mode="floor")

    def localize_routes(
        self, global_indices: torch.Tensor, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return local indices, zeroed non-local scores, and a locality mask."""
        if global_indices.shape != scores.shape:
            raise ValueError("Expert indices and scores must have the same shape")
        local_mask = self.owner(global_indices) == self.ep_rank
        sentinel = self.local_count
        local_indices = (global_indices - self.start).masked_fill(~local_mask, sentinel)
        local_scores = scores.masked_fill(~local_mask, 0)
        return local_indices, local_scores, local_mask


def local_expert_range(num_experts: int, ep_size: int, ep_rank: int) -> range:
    return ExpertPartition(num_experts, ep_size, ep_rank).global_experts


def verify_expert_modules(model, partition: ExpertPartition) -> None:
    expert_modules = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".mlp.experts") and hasattr(module, "num_experts")
    ]
    if not expert_modules:
        raise RuntimeError("Expert-parallel plan found no expert modules")
    mismatched = [
        f"{name}={module.num_experts}"
        for name, module in expert_modules
        if module.num_experts != partition.local_count
    ]
    if mismatched:
        raise RuntimeError(
            f"Expected {partition.local_count} local experts per rank, got "
            + ", ".join(mismatched)
        )


class _AllReduceBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.contiguous().clone()
        with collective_profile(
            "ep_all_reduce",
            value=gradient,
            mode="replicated_token_grad",
            detail="expert_boundary",
        ):
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=ctx.group)
        return gradient, None


class _AllReduceForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        result = value.contiguous().clone()
        with collective_profile(
            "ep_all_reduce",
            value=result,
            mode="replicated_token_output",
            detail="expert_boundary",
        ):
            dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class ReplicatedTokenExpertParallel:
    """Install replicated-token EP routing, gradient, and combine boundaries."""

    def __init__(self, partition: ExpertPartition, group):
        self.partition = partition
        self.group = group
        self._handles = []

    def _router_output_hook(self, module, inputs, outputs):
        router_logits, scores, global_indices, *extra = outputs
        # The locality mask must run before the backward all-reduce. Otherwise
        # each rank masks the already-reduced gradient a second time and the
        # replicated router receives only its locally owned expert routes.
        scores = _AllReduceBackward.apply(scores, self.group)
        local_indices, local_scores, _ = self.partition.localize_routes(
            global_indices, scores
        )
        return router_logits, local_scores, local_indices, *extra

    def _experts_input_hook(self, module, inputs):
        hidden_states, local_indices, routing_weights = inputs
        return (
            _AllReduceBackward.apply(hidden_states, self.group),
            local_indices,
            routing_weights,
        )

    def _experts_output_hook(self, module, inputs, output):
        return _AllReduceForward.apply(output, self.group)

    def apply(self, model) -> None:
        if self._handles:
            raise RuntimeError("Expert parallel hooks are already installed")
        routers = []
        experts = []
        for name, module in model.named_modules():
            if name.endswith(".mlp.gate"):
                routers.append(module)
            elif name.endswith(".mlp.experts"):
                experts.append(module)
        if not routers or len(routers) != len(experts):
            raise RuntimeError(
                f"Expected paired MoE routers and experts, got "
                f"{len(routers)} routers and {len(experts)} expert modules"
            )
        for router in routers:
            self._handles.append(router.register_forward_hook(self._router_output_hook))
        for expert_module in experts:
            self._handles.append(
                expert_module.register_forward_pre_hook(self._experts_input_hook)
            )
            self._handles.append(
                expert_module.register_forward_hook(self._experts_output_hook)
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class SequenceParallelExpertParallel:
    """Route local SP tokens to expert owners with a reversible all-to-all."""

    def __init__(self, partition: ExpertPartition, group):
        self.partition = partition
        self.group = group
        self.dispatcher = TokenDispatcher(partition, group)
        self._handles = []
        self._states = {}

    def _experts_input_hook(self, module, inputs):
        hidden_states, global_indices, routing_weights = inputs
        hidden, indices, weights, state = self.dispatcher.dispatch(
            hidden_states, global_indices, routing_weights
        )
        self._states.setdefault(id(module), []).append(state)
        return hidden, indices, weights

    def _experts_output_hook(self, module, inputs, output):
        states = self._states.get(id(module))
        if not states:
            raise RuntimeError("Expert output has no matching token dispatch")
        return self.dispatcher.combine(output, states.pop())

    def apply(self, model) -> None:
        if self._handles:
            raise RuntimeError("Expert parallel hooks are already installed")
        experts = [
            module
            for name, module in model.named_modules()
            if name.endswith(".mlp.experts")
        ]
        if not experts:
            raise RuntimeError("Sequence-parallel EP found no expert modules")
        for module in experts:
            self._handles.append(
                module.register_forward_pre_hook(self._experts_input_hook)
            )
            self._handles.append(
                module.register_forward_hook(self._experts_output_hook)
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._states.clear()

from __future__ import annotations

import torch
import torch.distributed as dist


def _all_gather_contiguous(output, value, group) -> None:
    operation = getattr(dist, "all_gather_single", None)
    if operation is None:
        operation = dist.all_gather_into_tensor
    operation(output, value, group=group)


def _reduce_scatter_contiguous(output, value, group) -> None:
    operation = getattr(dist, "reduce_scatter_single", None)
    if operation is None:
        operation = dist.reduce_scatter_tensor
    operation(output, value, group=group)


def _canonical_dim(tensor: torch.Tensor, dim: int) -> int:
    return dim if dim >= 0 else tensor.ndim + dim


def _move_dim_to_front(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor.movedim(_canonical_dim(tensor, dim), 0).contiguous()


def _restore_front_dim(tensor: torch.Tensor, dim: int, ndim: int) -> torch.Tensor:
    target = dim if dim >= 0 else ndim + dim
    return tensor.movedim(0, target).contiguous()


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, dim):
        ctx.group = group
        ctx.dim = dim
        ctx.world_size = dist.get_world_size(group)
        if ctx.world_size == 1:
            return value
        front = _move_dim_to_front(value, dim)
        gathered = torch.empty(
            (front.shape[0] * ctx.world_size, *front.shape[1:]),
            dtype=front.dtype,
            device=front.device,
        )
        _all_gather_contiguous(gathered, front, group)
        return _restore_front_dim(gathered, dim, value.ndim)

    @staticmethod
    def backward(ctx, gradient):
        if ctx.world_size == 1:
            return gradient, None, None
        front = _move_dim_to_front(gradient, ctx.dim)
        output = torch.empty(
            (front.shape[0] // ctx.world_size, *front.shape[1:]),
            dtype=front.dtype,
            device=front.device,
        )
        _reduce_scatter_contiguous(output, front, ctx.group)
        return _restore_front_dim(output, ctx.dim, gradient.ndim), None, None


class _ReduceScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, dim):
        ctx.group = group
        ctx.dim = dim
        ctx.world_size = dist.get_world_size(group)
        if ctx.world_size == 1:
            return value
        dim = _canonical_dim(value, dim)
        if value.shape[dim] % ctx.world_size:
            raise ValueError(
                f"Sequence dimension {value.shape[dim]} must be divisible by "
                f"sequence-parallel size {ctx.world_size}"
            )
        front = _move_dim_to_front(value, dim)
        output = torch.empty(
            (front.shape[0] // ctx.world_size, *front.shape[1:]),
            dtype=front.dtype,
            device=front.device,
        )
        _reduce_scatter_contiguous(output, front, group)
        return _restore_front_dim(output, dim, value.ndim)

    @staticmethod
    def backward(ctx, gradient):
        if ctx.world_size == 1:
            return gradient, None, None
        front = _move_dim_to_front(gradient, ctx.dim)
        gathered = torch.empty(
            (front.shape[0] * ctx.world_size, *front.shape[1:]),
            dtype=front.dtype,
            device=front.device,
        )
        _all_gather_contiguous(gathered, front, ctx.group)
        return _restore_front_dim(gathered, ctx.dim, gradient.ndim), None, None


class _ScatterSequence(torch.autograd.Function):
    """Split a replicated tensor; gather its gradient back in backward."""

    @staticmethod
    def forward(ctx, value, group, dim):
        ctx.group = group
        ctx.dim = dim
        ctx.world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        dim = _canonical_dim(value, dim)
        if value.shape[dim] % ctx.world_size:
            raise ValueError(
                f"Sequence dimension {value.shape[dim]} must be divisible by "
                f"sequence-parallel size {ctx.world_size}"
            )
        return value.chunk(ctx.world_size, dim=dim)[rank].contiguous()

    @staticmethod
    def backward(ctx, gradient):
        if ctx.world_size == 1:
            return gradient, None, None
        front = _move_dim_to_front(gradient, ctx.dim)
        gathered = torch.empty(
            (front.shape[0] * ctx.world_size, *front.shape[1:]),
            dtype=front.dtype,
            device=front.device,
        )
        _all_gather_contiguous(gathered, front, ctx.group)
        return _restore_front_dim(gathered, ctx.dim, gradient.ndim), None, None


def gather_sequence(value: torch.Tensor, group, dim: int = 1) -> torch.Tensor:
    return _GatherSequence.apply(value, group, dim)


def reduce_scatter_sequence(
    value: torch.Tensor, group, dim: int = 1
) -> torch.Tensor:
    return _ReduceScatterSequence.apply(value, group, dim)


def scatter_sequence(value: torch.Tensor, group, dim: int = 1) -> torch.Tensor:
    return _ScatterSequence.apply(value, group, dim)


class SequenceParallelRuntime:
    """Keep Qwen residual activations sequence-sharded over the TP group."""

    def __init__(self, group, *, dense_mlp: bool):
        self.group = group
        self.dense_mlp = dense_mlp
        self._handles = []

    def _first_layer_input(self, module, args, kwargs):
        if args:
            return (scatter_sequence(args[0], self.group), *args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = scatter_sequence(
            kwargs["hidden_states"], self.group
        )
        return args, kwargs

    def _gather_hidden(self, module, args, kwargs):
        if args:
            return (gather_sequence(args[0], self.group), *args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = gather_sequence(
            kwargs["hidden_states"], self.group
        )
        return args, kwargs

    def _gather_linear_input(self, module, args):
        return (gather_sequence(args[0], self.group), *args[1:])

    def apply(self, model) -> None:
        if self._handles:
            raise RuntimeError("Sequence-parallel hooks are already installed")
        layers = list(model.model.layers)
        if not layers:
            raise RuntimeError("Sequence parallelism requires decoder layers")
        self._handles.append(
            layers[0].register_forward_pre_hook(
                self._first_layer_input, with_kwargs=True
            )
        )
        for layer in layers:
            self._handles.append(
                layer.self_attn.register_forward_pre_hook(
                    self._gather_hidden, with_kwargs=True
                )
            )
            if self.dense_mlp:
                self._handles.append(
                    layer.mlp.register_forward_pre_hook(self._gather_linear_input)
                )
        self._handles.append(
            model.lm_head.register_forward_pre_hook(self._gather_linear_input)
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class _AllReduceForwardIdentityBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        result = value.contiguous().clone()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def sequence_parallel_load_balancing_loss(
    router_logits: tuple[torch.Tensor, ...] | None,
    *,
    num_experts: int,
    top_k: int,
    group,
) -> torch.Tensor | None:
    """Switch auxiliary loss over the global SP token population."""
    if not router_logits:
        return None
    logits = torch.cat(tuple(router_logits), dim=0)
    probabilities = torch.softmax(logits, dim=-1)
    selected = torch.topk(probabilities, top_k, dim=-1).indices
    assignments = torch.nn.functional.one_hot(selected, num_experts).float()

    counts = assignments.sum(dim=0)
    probability_sum = probabilities.sum(dim=0)
    token_count = torch.tensor(
        logits.shape[0], dtype=torch.float32, device=logits.device
    )
    dist.all_reduce(counts, group=group)
    dist.all_reduce(token_count, group=group)
    probability_sum = _AllReduceForwardIdentityBackward.apply(
        probability_sum, group
    )
    tokens_per_expert = counts / token_count
    probability_per_expert = probability_sum / token_count
    return num_experts * torch.sum(
        tokens_per_expert * probability_per_expert.unsqueeze(0)
    )

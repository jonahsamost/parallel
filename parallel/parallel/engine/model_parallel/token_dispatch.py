from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ...profiling import collective_profile


class _VariableAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, send_splits, receive_splits, group, mode, detail):
        ctx.send_splits = tuple(send_splits)
        ctx.receive_splits = tuple(receive_splits)
        ctx.group = group
        ctx.mode = mode
        ctx.detail = detail
        output = torch.empty(
            (sum(ctx.receive_splits), *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
        )
        with collective_profile(
            "ep_all_to_all",
            value=value,
            mode=mode,
            detail=detail,
        ):
            dist.all_to_all_single(
                output,
                value.contiguous(),
                output_split_sizes=list(ctx.receive_splits),
                input_split_sizes=list(ctx.send_splits),
                group=group,
            )
        return output

    @staticmethod
    def backward(ctx, gradient):
        output = torch.empty(
            (sum(ctx.send_splits), *gradient.shape[1:]),
            dtype=gradient.dtype,
            device=gradient.device,
        )
        with collective_profile(
            "ep_all_to_all",
            value=gradient,
            mode=f"{ctx.mode}_grad",
            detail=ctx.detail,
        ):
            dist.all_to_all_single(
                output,
                gradient.contiguous(),
                output_split_sizes=list(ctx.send_splits),
                input_split_sizes=list(ctx.receive_splits),
                group=ctx.group,
            )
        return output, None, None, None, None, None


def variable_all_to_all(
    value, send_splits, receive_splits, group, *, mode="", detail=""
):
    return _VariableAllToAll.apply(
        value, send_splits, receive_splits, group, mode, detail
    )


@dataclass
class DispatchState:
    send_splits: tuple[int, ...]
    receive_splits: tuple[int, ...]
    source_tokens: torch.Tensor
    num_source_tokens: int


class TokenDispatcher:
    """Dispatch top-k routes to expert owners and restore source-token order."""

    def __init__(self, partition, group):
        self.partition = partition
        self.group = group
        self.world_size = dist.get_world_size(group)

    def _exchange_splits(self, destinations: torch.Tensor):
        # sum(receive_counts) is how many tokens get routed to us
        send_counts = torch.bincount(
            destinations, minlength=self.world_size
        ).to(dtype=torch.int64)
        receive_counts = torch.empty_like(send_counts)
        with collective_profile(
            "ep_all_to_all",
            value=send_counts,
            mode="split_counts",
            detail="route_counts",
        ):
            dist.all_to_all_single(receive_counts, send_counts, group=self.group)
        return tuple(send_counts.tolist()), tuple(receive_counts.tolist())

    def dispatch(self, hidden_states, global_indices, scores):
        if global_indices.shape != scores.shape:
            raise ValueError("Expert indices and scores must have the same shape")
        tokens, top_k = global_indices.shape
        flat_experts = global_indices.reshape(-1)
        flat_scores = scores.reshape(-1)
        source_tokens = (
            torch.arange(tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(tokens, top_k)
            .reshape(-1)
        )
        # destinations contains the rank that owns each route's selected expert
        destinations = self.partition.owner(flat_experts)
        # order returns the indices that would sort based on rank
        order = torch.argsort(destinations, stable=True)
        # group every route for rank N together, followed by every route for rank N+1, etc
        destinations = destinations[order]
        source_tokens = source_tokens[order]
        # how many tokens are routed from/to each rank
        send_splits, receive_splits = self._exchange_splits(destinations)

        send_hidden = hidden_states.index_select(0, source_tokens)
        send_experts = (flat_experts[order] - destinations * self.partition.local_count)
        send_scores = flat_scores[order]
        received_hidden = variable_all_to_all(
            send_hidden,
            send_splits,
            receive_splits,
            self.group,
            mode="dispatch_hidden",
            detail="routes",
        )
        received_experts = variable_all_to_all(
            send_experts,
            send_splits,
            receive_splits,
            self.group,
            mode="dispatch_experts",
            detail="routes",
        )
        received_scores = variable_all_to_all(
            send_scores,
            send_splits,
            receive_splits,
            self.group,
            mode="dispatch_scores",
            detail="routes",
        )
        state = DispatchState(
            send_splits, receive_splits, source_tokens, tokens
        )
        return (
            received_hidden,
            received_experts.unsqueeze(-1),
            received_scores.unsqueeze(-1),
            state,
        )

    def combine(self, expert_output: torch.Tensor, state: DispatchState):
        returned = variable_all_to_all(
            expert_output,
            state.receive_splits,
            state.send_splits,
            self.group,
            mode="combine_hidden",
            detail="routes",
        )
        result = torch.zeros(
            (state.num_source_tokens, *returned.shape[1:]),
            dtype=returned.dtype,
            device=returned.device,
        )
        return result.index_add(0, state.source_tokens, returned)

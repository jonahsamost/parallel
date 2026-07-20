from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import loss_parallel


def loss_parallel_context(enabled: bool):
    return loss_parallel() if enabled else nullcontext()


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        stride[index] = stride[index + 1] * shape[index + 1]
    return tuple(stride)


def vocab_parallel_cross_entropy(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    tp_mesh,
    vocab_size: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross entropy without gathering vocabulary-sharded logits."""
    if isinstance(local_logits, DTensor):
        logits = local_logits
    else:
        if vocab_size % tp_mesh.size():
            raise ValueError(
                f"Vocabulary size {vocab_size} is not divisible by TP size {tp_mesh.size()}"
            )
        expected_local_vocab = vocab_size // tp_mesh.size()
        if local_logits.shape[-1] != expected_local_vocab:
            raise RuntimeError(
                "Expected a vocabulary shard with last dimension "
                f"{expected_local_vocab}, got {local_logits.shape[-1]}"
            )
        global_shape = (*local_logits.shape[:-1], vocab_size)
        logits = DTensor.from_local(
            local_logits,
            device_mesh=tp_mesh,
            placements=(Shard(-1),),
            run_check=False,
            shape=torch.Size(global_shape),
            stride=_contiguous_stride(global_shape),
        )

    """
    When thinking about parallel loss: 
        You will have inputs as [T, H] from your transformer block and weights
        [vocab_size, H] for you lm_head. Each rank holds _full_ [T, H] and sharding occurs
        over vocab_size, so each rank holds [T, vocab_size / tp_rank_size] as input to loss.
        `loss_parallel` hands the dist comms for all_reduce for sum/max. Sequence parallelism
        would allow each rank to just hold [T / tp_size, vocab_size] to eliminate the dist comms
        but would force each rank to hold the full [vocab_size, H] weights.
    """
    with loss_parallel():
        loss = F.cross_entropy(
            logits.float().view(-1, vocab_size),
            labels.reshape(-1).to(local_logits.device),
            reduction=reduction,
        )
    return loss.to_local() if isinstance(loss, DTensor) else loss

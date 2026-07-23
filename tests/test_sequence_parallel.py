import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from parallel.parallel.engine.model_parallel import ExpertPartition, TokenDispatcher
from parallel.parallel.profiling import (
    collective_phase,
    configure_collective_profiling,
    finish_collective_profile_step,
    start_collective_profile_step,
)


def _run_zero_receive_dispatch(rank, world_size, rendezvous_file):
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        configure_collective_profiling(enabled=True, device="cpu")
        start_collective_profile_step()
        partition = ExpertPartition(num_experts=4, ep_size=2, ep_rank=rank)
        dispatcher = TokenDispatcher(partition, dist.group.WORLD)
        hidden = (
            torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank * 10
        ).requires_grad_()
        scores = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4]], requires_grad=True
        )
        # Every route belongs to rank zero. Rank one therefore receives no
        # expert work while still sending routes and receiving combined output.
        indices = torch.tensor([[0, 1], [1, 0]])

        with collective_phase("forward"):
            received, local_indices, received_scores, state = dispatcher.dispatch(
                hidden, indices, scores
            )
            assert local_indices.shape == (8 if rank == 0 else 0, 1)
            expert_output = received * received_scores
            combined = dispatcher.combine(expert_output, state)
            torch.testing.assert_close(combined, hidden)

        with collective_phase("backward"):
            combined.square().sum().backward()
        torch.testing.assert_close(hidden.grad, 2 * hidden)
        torch.testing.assert_close(
            scores.grad,
            2
            * hidden.detach().square().sum(dim=-1, keepdim=True).expand_as(scores),
        )
        profile_lines = finish_collective_profile_step()
        assert any("op=ep_all_to_all" in line for line in profile_lines)
        assert any("phase=forward" in line for line in profile_lines)
        assert any("phase=backward" in line for line in profile_lines)
    finally:
        configure_collective_profiling(enabled=False)
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_token_dispatch_handles_a_rank_receiving_zero_tokens(tmp_path):
    mp.spawn(
        _run_zero_receive_dispatch,
        args=(2, tmp_path / "zero-receive-rendezvous"),
        nprocs=2,
        join=True,
    )

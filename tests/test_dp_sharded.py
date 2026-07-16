from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from parallel.parallel.engine.dp_sharded import FSDPWrapper
from parallel.parallel.engine.engine import ParallelEngine


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 8), nn.Linear(8, 4)])
        self.head = nn.Linear(4, 2)
        for module in [*self.layers, self.head]:
            module.weight.requires_grad_(False)

    def forward(self, inputs):
        hidden = self.layers[0](inputs).relu()
        hidden = self.layers[1](hidden).relu()
        return {"logits": self.head(hidden)}


def _run_fsdp_parity(rank, world_size, rendezvous_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(1234)
        reference = TinyModel()
        model = TinyModel()
        model.load_state_dict(reference.state_dict())
        if rank > 0:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(rank)

        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_shard",))
        pconfig = SimpleNamespace(dp_shard_size=world_size, device_mesh=mesh)
        wrapper = FSDPWrapper(model, pconfig, device=torch.device("cpu"))
        wrapper.shard_model()

        optimizer_params = wrapper.get_optimizer_params()
        optimizer_param_ids = {id(parameter) for parameter in optimizer_params}
        assert optimizer_params
        assert all(parameter.requires_grad for parameter in optimizer_params)
        assert all(parameter.numel() == 0 for parameter in model.parameters())
        assert all(
            id(unit.flat_shard) not in optimizer_param_ids
            for unit in wrapper.units
            if not unit.requires_grad
        )

        optimizer = torch.optim.SGD(optimizer_params, lr=0.05)
        reference_optimizer = torch.optim.SGD(
            [parameter for parameter in reference.parameters() if parameter.requires_grad],
            lr=0.05,
        )
        local_microbatches = [
            (torch.arange(12, dtype=torch.float32).view(3, 4) + rank * 24) / 10,
            (torch.arange(12, 24, dtype=torch.float32).view(3, 4) + rank * 24) / 10,
        ]

        for inputs in local_microbatches:
            model(inputs)["logits"].square().mean().div_(len(local_microbatches)).backward()
            wrapper.finalize_backward()

        for data_rank in range(world_size):
            reference_microbatches = [
                (torch.arange(12, dtype=torch.float32).view(3, 4) + data_rank * 24) / 10,
                (torch.arange(12, 24, dtype=torch.float32).view(3, 4) + data_rank * 24) / 10,
            ]
            for inputs in reference_microbatches:
                reference_loss = reference(inputs)["logits"].square().mean()
                reference_loss.div_(len(reference_microbatches) * world_size).backward()

        optimizer.step()
        reference_optimizer.step()

        actual_state = wrapper.state_dict()
        expected_state = reference.state_dict()
        for name, expected in expected_state.items():
            torch.testing.assert_close(actual_state[name], expected)

        buffer_model = nn.Module()
        buffer_model.register_buffer("value", torch.tensor(float(rank)))
        engine = ParallelEngine.__new__(ParallelEngine)
        engine.pconfig = SimpleNamespace(device_mesh=mesh)
        engine.model = buffer_model
        engine.sync_buffers()
        torch.testing.assert_close(buffer_model.value, torch.tensor(0.0))
    finally:
        dist.destroy_process_group()


def _run_unused_grad_sync(rank, world_size, rendezvous_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_replicate",))
        engine = ParallelEngine.__new__(ParallelEngine)
        engine.pconfig = SimpleNamespace(dp_replicate_size=world_size, device_mesh=mesh)
        engine.device = torch.device("cpu")
        engine.optimizer_params = [
            nn.Parameter(torch.tensor([1.0])),
            nn.Parameter(torch.tensor([2.0])),
        ]
        engine.optimizer_params[rank].grad = torch.ones(1)

        engine._average_gradients()

        for param in engine.optimizer_params:
            torch.testing.assert_close(param.grad, torch.tensor([0.5]))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_fsdp_matches_unsharded_with_gradient_accumulation(tmp_path):
    world_size = 2
    rendezvous_file = tmp_path / "rendezvous"
    mp.spawn(
        _run_fsdp_parity,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_replicated_dp_synchronizes_rank_local_unused_gradients(tmp_path):
    world_size = 2
    rendezvous_file = tmp_path / "rendezvous"
    mp.spawn(
        _run_unused_grad_sync,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )

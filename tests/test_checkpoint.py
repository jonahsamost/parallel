import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from parallel.parallel.engine.checkpoint import CheckpointManager
from parallel.parallel.engine.dp_sharded import FSDPWrapper


class CheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, 4, bias=False)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.output = nn.Linear(4, 4, bias=False)
        self.output.weight = self.input.weight
        self.register_buffer("running_value", torch.tensor(3.0))

    def forward(self, inputs):
        hidden = self.input(inputs)
        for layer in self.layers:
            hidden = layer(hidden).relu()
        return self.output(hidden)


def _assert_optimizer_states_equal(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id, expected_state in expected["state"].items():
        actual_state = actual["state"][parameter_id]
        assert actual_state.keys() == expected_state.keys()
        for name, expected_value in expected_state.items():
            actual_value = actual_state[name]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(actual_value.cpu(), expected_value.cpu())
            else:
                assert actual_value == expected_value


def _run_checkpoint_round_trip(rank, world_size, rendezvous_file, checkpoint_root):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(2025)
        model = CheckpointModel()
        reference_names = list(model.state_dict().keys())
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_shard",))
        pconfig = SimpleNamespace(
            dp_replicate_size=1,
            dp_shard_size=world_size,
            device_mesh=mesh,
        )
        wrapper = FSDPWrapper(model, pconfig, device=torch.device("cpu"))
        wrapper.shard_model()
        optimizer = torch.optim.AdamW(wrapper.get_optimizer_params(), lr=0.01)
        engine = SimpleNamespace(
            fsdp_wrapper=wrapper,
            optimizer=optimizer,
            grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
            scheduler=None,
            pconfig=pconfig,
            device=torch.device("cpu"),
            cfg=None,
        )
        checkpoint = CheckpointManager(engine)

        inputs = torch.arange(8, dtype=torch.float32).view(2, 4).add(rank)
        loss = model(inputs).square().mean()
        wrapper.prepare_backward(loss)
        loss.backward()
        wrapper.finalize_backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        full_state = checkpoint.full_state_dict()
        assert all(unit._full_params_buf is None for unit in wrapper.units)
        if rank == 0:
            assert list(full_state.keys()) == reference_names
            assert all(value.device.type == "cpu" for value in full_state.values())
            assert full_state["input.weight"] is full_state["output.weight"]
            expected_full_state = {
                name: value.clone() for name, value in full_state.items()
            }
        else:
            assert full_state is None
            expected_full_state = None

        full_path = checkpoint_root / "full-model.pt"
        checkpoint.save_full_model(full_path)
        for parameter in wrapper.get_sharded_params():
            parameter.to_local().detach().zero_()
        checkpoint.load_full_model(full_path)
        restored_full_state = checkpoint.full_state_dict()
        if rank == 0:
            for name, expected in expected_full_state.items():
                torch.testing.assert_close(restored_full_state[name], expected)

        saved_shards = [
            parameter.to_local().detach().clone()
            for parameter in wrapper.get_sharded_params()
        ]
        saved_optimizer = optimizer.state_dict()
        saved_buffer = model.running_value.detach().clone()
        saved_rng = torch.get_rng_state()
        saved_python_rng = random.getstate()
        saved_numpy_rng = np.random.get_state()
        expected_random = torch.rand(4)
        expected_python_random = random.random()
        expected_numpy_random = np.random.rand(4)
        torch.set_rng_state(saved_rng)
        random.setstate(saved_python_rng)
        np.random.set_state(saved_numpy_rng)

        checkpoint_path = checkpoint_root / "training-state"
        checkpoint.save(
            checkpoint_path,
            step=17,
            dataloader_state={"rank": rank, "position": rank + 5},
            eval_dataloader_state={"rank": rank, "position": rank + 9},
            metadata={"purpose": "round-trip"},
        )

        for parameter in wrapper.get_sharded_params():
            parameter.to_local().detach().fill_(rank + 100)
        optimizer.state.clear()
        model.running_value.add_(10)
        torch.manual_seed(rank + 99)
        random.seed(rank + 99)
        np.random.seed(rank + 99)

        resume = checkpoint.load(checkpoint_path)
        assert resume["step"] == 17
        assert resume["dataloader_state"] == {"rank": rank, "position": rank + 5}
        assert resume["eval_dataloader_state"] == {
            "rank": rank,
            "position": rank + 9,
        }
        assert resume["metadata"] == {"purpose": "round-trip"}
        for actual, expected in zip(wrapper.get_sharded_params(), saved_shards):
            torch.testing.assert_close(actual.to_local(), expected)
        _assert_optimizer_states_equal(optimizer.state_dict(), saved_optimizer)
        torch.testing.assert_close(model.running_value, saved_buffer)
        torch.testing.assert_close(torch.rand(4), expected_random)
        assert random.random() == expected_python_random
        np.testing.assert_array_equal(np.random.rand(4), expected_numpy_random)

        final_state = checkpoint.full_state_dict()
        if rank == 0:
            for name, expected in expected_full_state.items():
                torch.testing.assert_close(final_state[name], expected)
            assert (checkpoint_path / "COMPLETE").is_file()
            assert (checkpoint_path / "manifest.json").is_file()

        model_path = checkpoint_path / checkpoint._model_file_name(rank)
        model_payload = torch.load(model_path, map_location="cpu", weights_only=False)
        del model_payload["model"]["non_parameter_state"]["running_value"]
        torch.save(model_payload, model_path)
        dist.barrier()
        with pytest.raises(RuntimeError, match="Missing non-parameter key"):
            checkpoint.load(checkpoint_path)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_full_and_sharded_checkpoint_round_trip(tmp_path):
    world_size = 2
    rendezvous_file = tmp_path / "rendezvous"
    mp.spawn(
        _run_checkpoint_round_trip,
        args=(world_size, rendezvous_file, tmp_path),
        nprocs=world_size,
        join=True,
    )


def test_unsharded_training_checkpoint_round_trip(tmp_path):
    torch.manual_seed(1234)
    model = CheckpointModel()
    pconfig = SimpleNamespace(
        dp_replicate_size=1,
        dp_shard_size=1,
        device_mesh=None,
    )
    wrapper = FSDPWrapper(model, pconfig, device=torch.device("cpu"))
    wrapper.shard_model()
    optimizer = torch.optim.AdamW(wrapper.get_optimizer_params(), lr=0.01)
    loss = model(torch.arange(8, dtype=torch.float32).view(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    engine = SimpleNamespace(
        fsdp_wrapper=wrapper,
        optimizer=optimizer,
        grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
        scheduler=None,
        pconfig=pconfig,
        device=torch.device("cpu"),
        cfg=None,
    )
    checkpoint = CheckpointManager(engine)
    expected_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_optimizer = optimizer.state_dict()

    checkpoint_path = tmp_path / "unsharded"
    checkpoint.save(
        checkpoint_path,
        step=8,
        dataloader_state={"position": 10},
        eval_dataloader_state={"position": 4},
    )
    for parameter in model.parameters():
        parameter.detach().zero_()
    optimizer.state.clear()

    resume = checkpoint.load(checkpoint_path)
    assert resume["step"] == 8
    assert resume["dataloader_state"] == {"position": 10}
    assert resume["eval_dataloader_state"] == {"position": 4}
    for name, expected in expected_model.items():
        torch.testing.assert_close(model.state_dict()[name], expected)
    _assert_optimizer_states_equal(optimizer.state_dict(), expected_optimizer)

    rank_path = checkpoint_path / checkpoint._rank_file_name(0)
    rank_payload = torch.load(rank_path, map_location="cpu", weights_only=False)
    rank_payload["format_version"] = -1
    torch.save(rank_payload, rank_path)
    with pytest.raises(RuntimeError, match="unsupported format version"):
        checkpoint.load(checkpoint_path)

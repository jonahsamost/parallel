from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from parallel.parallel.engine.dp_sharded import FSDPWrapper
from parallel.parallel.engine.engine import ParallelEngine
from parallel.parallel.engine._dp_param_unit import DPParamUnit


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


class RankConditionalModel(nn.Module):
    """Runs every unit in forward but selects rank-local branches for backward."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(4, 4),
                nn.Linear(4, 4),
                nn.Linear(4, 4),  # Globally unused output.
                nn.Linear(4, 2),  # Shared tail, and first unit ready in backward.
            ]
        )

    def forward(self, inputs, branch):
        branch_outputs = [self.layers[0](inputs), self.layers[1](inputs)]
        self.layers[2](inputs)
        return self.layers[3](branch_outputs[branch]).square().mean()


class SparseUnitModel(nn.Module):
    """One FSDP unit with only one locally selected parameter subset."""

    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(4, 2), nn.Linear(4, 2)])

    def forward(self, inputs, expert):
        return self.experts[expert](inputs).square().mean()


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
            loss = model(inputs)["logits"].square().mean().div_(len(local_microbatches))
            wrapper.prepare_backward(loss)
            loss.backward()
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
        if rank == 0:
            for name, expected in expected_state.items():
                torch.testing.assert_close(actual_state[name], expected)
        else:
            assert actual_state is None

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


def _run_fsdp_rank_local_unused(rank, world_size, rendezvous_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(4321)
        reference = RankConditionalModel()
        model = RankConditionalModel()
        model.load_state_dict(reference.state_dict())

        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_shard",))
        pconfig = SimpleNamespace(dp_shard_size=world_size, device_mesh=mesh)
        wrapper = FSDPWrapper(model, pconfig, device=torch.device("cpu"))
        wrapper.shard_model()

        optimizer = torch.optim.SGD(wrapper.get_optimizer_params(), lr=0.1, weight_decay=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, weight_decay=0.1)
        inputs = torch.arange(8, dtype=torch.float32).view(2, 4).div(10).add(rank)

        loss = model(inputs, rank)
        wrapper.prepare_backward(loss)
        loss.backward()
        wrapper.finalize_backward()

        # A unit unused on every rank must retain grad=None so weight decay does
        # not update it merely because we issued a zero-filled reduce-scatter.
        assert wrapper.units[2].flat_shard.grad is None

        for data_rank in range(world_size):
            reference_inputs = (
                torch.arange(8, dtype=torch.float32)
                .view(2, 4)
                .div(10)
                .add(data_rank)
            )
            reference(reference_inputs, data_rank).div(world_size).backward()

        optimizer.step()
        reference_optimizer.step()

        actual_state = wrapper.state_dict()
        if rank == 0:
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(actual_state[name], expected)
        else:
            assert actual_state is None
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


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_fsdp_synchronizes_rank_local_unused_gradients_in_fixed_order(tmp_path):
    world_size = 2
    rendezvous_file = tmp_path / "rendezvous"
    mp.spawn(
        _run_fsdp_rank_local_unused,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


def test_backward_reducer_drains_ready_units_in_reverse_order():
    calls = []
    units = []
    for index in range(3):
        unit = Mock(requires_grad=True, unit_index=index)
        unit.device = torch.device("cpu")
        unit.start_gradient_reduction.side_effect = (
            lambda current=index: calls.append(("start", current))
        )
        unit.local_gradient_participated = True
        unit.finish_gradient_reduction.side_effect = (
            lambda current=index: calls.append(("finish", current))
        )
        unit.accumulate_reduced_gradient.side_effect = (
            lambda *, accumulate, current=index: calls.append(
                ("accumulate", current, accumulate)
            )
        )
        units.append(unit)

    wrapper = FSDPWrapper.__new__(FSDPWrapper)
    wrapper.units = units
    wrapper._backward_units = []
    wrapper._backward_positions = {}
    wrapper._ready_backward_positions = set()
    wrapper._next_backward_position = -1
    wrapper._inflight_backward_positions = []
    wrapper._max_inflight_gradient_reductions = 2
    wrapper.dp_shard_group = Mock()
    wrapper._initialize_backward_reducer()
    wrapper._backward_prepared = True
    wrapper._ready_backward_positions.add(0)  # Discovered unused before backward.

    wrapper._on_unit_gradients_ready(units[1])
    assert calls == []
    wrapper._on_unit_gradients_ready(units[2])
    assert calls == [
        ("start", 2),
        ("start", 1),
        ("finish", 2),
        ("start", 0),
    ]

    with patch.object(dist, "all_reduce") as all_reduce:
        wrapper.finalize_backward()
    all_reduce.assert_called_once()
    assert calls == [
        ("start", 2),
        ("start", 1),
        ("finish", 2),
        ("start", 0),
        ("finish", 1),
        ("finish", 0),
        ("accumulate", 2, True),
        ("accumulate", 1, True),
        ("accumulate", 0, True),
    ]
    for unit in units:
        unit.reset_backward_state.assert_called_once_with()


def test_backward_hooks_launch_reduction_before_finalize_without_network():
    group = object()
    mesh = SimpleNamespace(get_group=Mock(return_value=group))

    def all_gather_into_tensor(output, input_, **kwargs):
        output.copy_(input_)

    def reduce_scatter_tensor(output, input_, **kwargs):
        output.copy_(input_)
        return Mock(wait=Mock())

    with (
        patch.object(dist, "get_world_size", return_value=1),
        patch.object(dist, "get_rank", return_value=0),
        patch.object(dist, "get_global_rank", return_value=0),
        patch.object(dist, "broadcast"),
        patch.object(dist, "all_gather_into_tensor", side_effect=all_gather_into_tensor),
        patch.object(dist, "reduce_scatter_tensor", side_effect=reduce_scatter_tensor),
        patch.object(dist, "all_reduce"),
    ):
        torch.manual_seed(2468)
        reference = RankConditionalModel()
        model = RankConditionalModel()
        model.load_state_dict(reference.state_dict())
        wrapper = FSDPWrapper(
            model,
            SimpleNamespace(dp_shard_size=2, device_mesh=mesh),
            device=torch.device("cpu"),
        )
        wrapper.shard_model()

        inputs = torch.arange(8, dtype=torch.float32).view(2, 4).div(10)
        loss = model(inputs, 0).div(2)
        wrapper.prepare_backward(loss)
        loss.backward()

        # Graph discovery pre-marks both unused units, so the shared tail and
        # selected branch drain the entire fixed schedule during autograd.
        assert all(unit._gradient_reduction_started for unit in wrapper.units)
        assert wrapper._next_backward_position == -1
        assert (
            len(wrapper._inflight_backward_positions)
            <= wrapper._max_inflight_gradient_reductions
        )

        wrapper.finalize_backward()
        assert wrapper.units[2].flat_shard.grad is None

        second_inputs = inputs.add(1)
        second_loss = model(second_inputs, 0).div(2)
        wrapper.prepare_backward(second_loss)
        second_loss.backward()
        wrapper.finalize_backward()
        assert wrapper.units[2].flat_shard.grad is None

        optimizer = torch.optim.SGD(wrapper.get_optimizer_params(), lr=0.1, weight_decay=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, weight_decay=0.1)
        reference(inputs, 0).div(2).backward()
        reference(second_inputs, 0).div(2).backward()
        optimizer.step()
        reference_optimizer.step()

        actual_state = wrapper.state_dict()
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(actual_state[name], expected)


def test_partially_used_unit_reduces_after_expected_subset_accumulates():
    group = object()
    mesh = SimpleNamespace(get_group=Mock(return_value=group))

    def all_gather_into_tensor(output, input_, **kwargs):
        output.copy_(input_)

    def reduce_scatter_tensor(output, input_, **kwargs):
        output.copy_(input_)
        return Mock(wait=Mock())

    with (
        patch.object(dist, "get_world_size", return_value=1),
        patch.object(dist, "get_rank", return_value=0),
        patch.object(dist, "get_global_rank", return_value=0),
        patch.object(dist, "broadcast"),
        patch.object(dist, "all_gather_into_tensor", side_effect=all_gather_into_tensor),
        patch.object(dist, "reduce_scatter_tensor", side_effect=reduce_scatter_tensor),
        patch.object(dist, "all_reduce"),
    ):
        torch.manual_seed(9753)
        reference = SparseUnitModel()
        model = SparseUnitModel()
        model.load_state_dict(reference.state_dict())
        wrapper = FSDPWrapper(
            model,
            SimpleNamespace(dp_shard_size=2, device_mesh=mesh),
            device=torch.device("cpu"),
        )
        wrapper.shard_model()

        inputs = torch.arange(8, dtype=torch.float32).view(2, 4).div(10)
        loss = model(inputs, 0)
        wrapper.prepare_backward(loss)
        unit = wrapper.units[0]
        expected_ids = {id(parameter) for parameter in model.experts[0].parameters()}
        assert unit._expected_grad_param_ids == expected_ids

        loss.backward()

        assert unit._gradient_reduction_started
        assert unit._full_params_buf is None
        assert all(parameter.grad is None for parameter in model.parameters())
        wrapper.finalize_backward()

        optimizer = torch.optim.SGD(wrapper.get_optimizer_params(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        reference(inputs, 0).backward()
        optimizer.step()
        reference_optimizer.step()

        actual_state = wrapper.state_dict()
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(actual_state[name], expected)


def test_prefetch_stream_waits_for_current_stream(monkeypatch):
    unit = DPParamUnit.__new__(DPParamUnit)
    unit._full_params_buf = None
    unit._prefetch_stream = Mock()
    unit._prefetch_event = None
    unit.device = torch.device("cuda:0")
    unit.all_gather = Mock()
    call_order = []
    unit._prefetch_stream.wait_stream.side_effect = (
        lambda stream: call_order.append("wait")
    )
    unit.all_gather.side_effect = lambda: call_order.append("gather")

    current_stream = Mock()
    prefetch_event = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=current_stream))
    monkeypatch.setattr(torch.cuda, "stream", Mock(return_value=nullcontext()))
    monkeypatch.setattr(torch.cuda, "Event", Mock(return_value=prefetch_event))

    unit.prefetch()

    unit._prefetch_stream.wait_stream.assert_called_once_with(current_stream)
    unit.all_gather.assert_called_once_with()
    prefetch_event.record.assert_called_once_with(unit._prefetch_stream)
    assert call_order == ["wait", "gather"]

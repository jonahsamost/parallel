from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

import parallel.parallel.engine._dp_param_unit as dp_param_unit_module
from parallel.parallel.engine.dp_sharded import FSDPWrapper
from parallel.parallel.engine.engine import ParallelEngine
from parallel.parallel.engine._dp_param_unit import (
    DPParamUnit,
    _all_gather_contiguous,
    _reduce_scatter_contiguous,
)


def test_contiguous_collectives_fall_back_to_pytorch_211_names(monkeypatch):
    gathered = object()
    reduced = object()
    all_gather = Mock(return_value=gathered)
    reduce_scatter = Mock(return_value=reduced)
    monkeypatch.setattr(dist, "all_gather_single", None, raising=False)
    monkeypatch.setattr(dist, "reduce_scatter_single", None, raising=False)
    monkeypatch.setattr(dist, "all_gather_into_tensor", all_gather)
    monkeypatch.setattr(dist, "reduce_scatter_tensor", reduce_scatter)

    output = torch.empty(2)
    input = torch.ones(1)
    assert _all_gather_contiguous(output, input, group="group") is gathered
    assert (
        _reduce_scatter_contiguous(
            output,
            input,
            op=dist.ReduceOp.SUM,
            group="group",
            async_op=True,
        )
        is reduced
    )
    all_gather.assert_called_once_with(output, input, group="group")
    reduce_scatter.assert_called_once_with(
        output,
        input,
        op=dist.ReduceOp.SUM,
        group="group",
        async_op=True,
    )


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


@contextmanager
def _single_rank_mesh(rendezvous_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=0,
        world_size=1,
    )
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))
    finally:
        dist.destroy_process_group()


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
        assert all(isinstance(parameter, DTensor) for parameter in model.parameters())
        assert all(parameter.placements == (Shard(0),) for parameter in model.parameters())
        assert optimizer_param_ids == {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        assert all(
            id(meta.sharded_param) not in optimizer_param_ids
            for unit in wrapper.units
            if not unit.requires_grad
            for meta in unit.param_metas
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


def _run_fsdp_rank_local_unused(
    rank,
    world_size,
    rendezvous_file,
    overlap_backward_reductions=True,
    activation_checkpoint=False,
):
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
        wrapper = FSDPWrapper(
            model,
            pconfig,
            find_unused_parameters=True,
            overlap_backward_reductions=overlap_backward_reductions,
            activation_checkpoint=activation_checkpoint,
            device=torch.device("cpu"),
        )
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
        assert all(
            meta.sharded_param.grad is None for meta in wrapper.units[2].param_metas
        )

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


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_fsdp_can_defer_rank_local_unused_gradient_reductions(tmp_path):
    world_size = 2
    rendezvous_file = tmp_path / "rendezvous"
    mp.spawn(
        _run_fsdp_rank_local_unused,
        args=(world_size, rendezvous_file, False),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_fsdp_rank_local_unused_with_activation_checkpointing(tmp_path):
    world_size = 2
    rendezvous_file = tmp_path / "rendezvous"
    mp.spawn(
        _run_fsdp_rank_local_unused,
        args=(world_size, rendezvous_file, True, True),
        nprocs=world_size,
        join=True,
    )


def test_backward_reducer_drains_ready_units_in_reverse_order():
    calls = []
    units = []
    for index in range(3):
        unit = Mock(requires_grad=True, unit_index=index)
        unit.device = torch.device("cpu")
        unit.param_metas = [Mock()]
        unit.start_gradient_reduction.side_effect = (
            lambda current=index: calls.append(("start", current))
        )
        unit.local_gradient_participation = [True]
        unit.finish_gradient_reduction.side_effect = (
            lambda current=index: calls.append(("finish", current))
        )
        unit.accumulate_reduced_gradients.side_effect = (
            lambda accumulate, current=index: calls.append(
                ("accumulate", current, list(accumulate))
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
    wrapper.find_unused_parameters = False
    wrapper.overlap_backward_reductions = True
    wrapper._globally_used_backward_positions = []
    wrapper._global_parameter_participation = {}
    wrapper._deferred_backward_positions = []
    wrapper._reduced_backward_positions = set()
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

    wrapper.finalize_backward()
    assert calls == [
        ("start", 2),
        ("start", 1),
        ("finish", 2),
        ("start", 0),
        ("finish", 1),
        ("finish", 0),
        ("accumulate", 2, [True]),
        ("accumulate", 1, [True]),
        ("accumulate", 0, [True]),
    ]
    for unit in units:
        unit.reset_backward_state.assert_called_once_with()


def test_backward_hooks_launch_reduction_before_finalize(tmp_path):
    with _single_rank_mesh(tmp_path / "hook-rendezvous") as mesh:
        torch.manual_seed(2468)
        reference = RankConditionalModel()
        model = RankConditionalModel()
        model.load_state_dict(reference.state_dict())
        wrapper = FSDPWrapper(
            model,
            SimpleNamespace(dp_shard_size=2, device_mesh=mesh),
            find_unused_parameters=True,
            device=torch.device("cpu"),
        )
        wrapper.shard_model()

        inputs = torch.arange(8, dtype=torch.float32).view(2, 4).div(10)
        loss = model(inputs, 0).div(2)
        wrapper.prepare_backward(loss)

        # Usage-aware scheduling starts only the globally next unit. It does
        # not bulk-materialize the selected branch before backward reaches it.
        assert wrapper.units[3]._full_params_buf is not None
        assert wrapper.units[0]._full_params_buf is None
        loss.backward()

        # Only the selected branch and shared tail enter the collective
        # schedule; globally unused units are skipped consistently.
        assert wrapper.units[0]._gradient_reduction_started
        assert not wrapper.units[1]._gradient_reduction_started
        assert not wrapper.units[2]._gradient_reduction_started
        assert wrapper.units[3]._gradient_reduction_started
        assert wrapper._next_backward_position == -1
        assert (
            len(wrapper._inflight_backward_positions)
            <= wrapper._max_inflight_gradient_reductions
        )

        wrapper.finalize_backward()
        assert all(
            meta.sharded_param.grad is None for meta in wrapper.units[2].param_metas
        )

        second_inputs = inputs.add(1)
        second_loss = model(second_inputs, 0).div(2)
        wrapper.prepare_backward(second_loss)
        second_loss.backward()
        wrapper.finalize_backward()
        assert all(
            meta.sharded_param.grad is None for meta in wrapper.units[2].param_metas
        )

        optimizer = torch.optim.SGD(wrapper.get_optimizer_params(), lr=0.1, weight_decay=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, weight_decay=0.1)
        reference(inputs, 0).div(2).backward()
        reference(second_inputs, 0).div(2).backward()
        optimizer.step()
        reference_optimizer.step()

        actual_state = wrapper.state_dict()
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(actual_state[name], expected)


def test_backward_reductions_can_be_deferred_until_finalize(tmp_path):
    with _single_rank_mesh(tmp_path / "deferred-rendezvous") as mesh:
        torch.manual_seed(8642)
        model = TinyModel()
        wrapper = FSDPWrapper(
            model,
            SimpleNamespace(dp_shard_size=2, device_mesh=mesh),
            overlap_backward_reductions=False,
            device=torch.device("cpu"),
        )
        wrapper.shard_model()

        inputs = torch.arange(12, dtype=torch.float32).view(3, 4).div(10)
        loss = model(inputs)["logits"].square().mean()
        wrapper.prepare_backward(loss)
        loss.backward()

        assert all(
            not unit._gradient_reduction_started
            for unit in wrapper._backward_units
        )
        assert all(
            unit._full_params_buf is None
            for unit in wrapper._backward_units
            if unit.reshard_after_forward
        )
        assert all(
            any(meta.parameter.grad is not None for meta in unit.param_metas)
            for unit in wrapper._backward_units
        )

        wrapper.finalize_backward()

        assert all(
            meta.sharded_param.grad is not None
            for unit in wrapper._backward_units
            for meta in unit.param_metas
        )


def test_partially_used_unit_reduces_after_expected_subset_accumulates(tmp_path):
    with _single_rank_mesh(tmp_path / "partial-rendezvous") as mesh:
        torch.manual_seed(9753)
        reference = SparseUnitModel()
        model = SparseUnitModel()
        model.load_state_dict(reference.state_dict())
        wrapper = FSDPWrapper(
            model,
            SimpleNamespace(dp_shard_size=2, device_mesh=mesh),
            find_unused_parameters=True,
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

        for meta in unit.param_metas:
            if id(meta.parameter) in expected_ids:
                assert meta.sharded_param.grad is not None
            else:
                assert meta.sharded_param.grad is None

        optimizer = torch.optim.SGD(
            wrapper.get_optimizer_params(), lr=0.1, weight_decay=0.1
        )
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.1, weight_decay=0.1
        )
        reference(inputs, 0).backward()
        optimizer.step()
        reference_optimizer.step()

        actual_state = wrapper.state_dict()
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(actual_state[name], expected)


def test_multi_parameter_unit_uses_one_packed_collective_each_way(
    tmp_path, monkeypatch
):
    with _single_rank_mesh(tmp_path / "bucket-rendezvous") as mesh:
        model = SparseUnitModel()
        wrapper = FSDPWrapper(
            model,
            SimpleNamespace(dp_shard_size=2, device_mesh=mesh),
            device=torch.device("cpu"),
        )
        wrapper.shard_model()
        unit = wrapper.units[0]
        assert len(unit.param_metas) > 1
        assert unit._local_param_bucket is not None
        bucket_storage = unit._local_param_bucket.untyped_storage().data_ptr()
        assert all(
            meta.padded_shard is not None
            and meta.padded_shard.untyped_storage().data_ptr() == bucket_storage
            for meta in unit.param_metas
        )

        original_all_gather = dp_param_unit_module._all_gather_contiguous
        original_reduce_scatter = dp_param_unit_module._reduce_scatter_contiguous
        all_gather_calls = []
        reduce_scatter_calls = []

        def tracked_all_gather(output, input, *, group):
            all_gather_calls.append((output.shape, input.shape))
            return original_all_gather(output, input, group=group)

        def tracked_reduce_scatter(output, input, *, op, group, async_op=False):
            reduce_scatter_calls.append((output.shape, input.shape))
            return original_reduce_scatter(
                output,
                input,
                op=op,
                group=group,
                async_op=async_op,
            )

        monkeypatch.setattr(
            dp_param_unit_module, "_all_gather_contiguous", tracked_all_gather
        )
        monkeypatch.setattr(
            dp_param_unit_module, "_reduce_scatter_contiguous", tracked_reduce_scatter
        )

        unit.all_gather()
        for meta in unit.param_metas:
            meta.parameter.grad = torch.ones_like(meta.parameter)
        unit.start_gradient_reduction()
        unit.finish_gradient_reduction()
        unit.accumulate_reduced_gradients([True] * len(unit.param_metas))
        first_grad_bucket = unit._accumulated_shard_grad_bucket_ref()
        assert first_grad_bucket is not None

        unit.reset_backward_state()
        unit.all_gather()
        for meta in unit.param_metas:
            meta.parameter.grad = torch.full_like(meta.parameter, 2)
        unit.start_gradient_reduction()
        assert unit._pending_shard_grad_bucket is first_grad_bucket
        unit.finish_gradient_reduction()
        unit.accumulate_reduced_gradients([True] * len(unit.param_metas))

        expected_shape = torch.Size([unit._bucket_numel])
        assert all_gather_calls == [
            (expected_shape, expected_shape),
            (expected_shape, expected_shape),
        ]
        assert reduce_scatter_calls == [
            (expected_shape, expected_shape),
            (expected_shape, expected_shape),
        ]
        for meta in unit.param_metas:
            torch.testing.assert_close(
                meta.sharded_param.grad._local_tensor,
                torch.full_like(meta.sharded_param.grad._local_tensor, 3),
            )


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
    unit.all_gather.side_effect = lambda **kwargs: call_order.append("gather")

    current_stream = Mock()
    prefetch_event = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=current_stream))
    monkeypatch.setattr(torch.cuda, "stream", Mock(return_value=nullcontext()))
    monkeypatch.setattr(torch.cuda, "Event", Mock(return_value=prefetch_event))

    unit.prefetch()

    unit._prefetch_stream.wait_stream.assert_called_once_with(current_stream)
    unit.all_gather.assert_called_once_with(mode="prefetch")
    prefetch_event.record.assert_called_once_with(unit._prefetch_stream)
    assert call_order == ["wait", "gather"]

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from omegaconf import OmegaConf
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from parallel.parallel.engine.engine import ParallelEngine
from parallel.parallel.engine.model_loading import load_causal_lm
from parallel.parallel.engine.model_parallel import (
    ExpertPartition,
    build_model_parallel_plan,
    build_qwen3_moe_plan,
)


def _tiny_qwen_config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 32,
        "attention_dropout": 0.0,
        "output_router_logits": True,
        "router_aux_loss_coef": 0.01,
        "use_cache": False,
    }
    values.update(overrides)
    return Qwen3MoeConfig(**values)


def _pconfig(mesh=None, size=2, *, sequence_parallel=False):
    return SimpleNamespace(
        tp_size=size,
        ep_size=size,
        expert_tp_size=1,
        sp_size=size if sequence_parallel else 1,
        dp_shard_size=1,
        model_parallel_enabled=size > 1,
        device_mesh=mesh,
        device_type="cpu",
        dp_replicate_size=1,
        dp_enabled=False,
    )


def _dense_pconfig(mesh=None, size=2, *, sequence_parallel=False):
    config = _pconfig(mesh, size, sequence_parallel=sequence_parallel)
    config.ep_size = 1
    return config


def _composed_pconfig(mesh, *, sequence_parallel=False):
    return SimpleNamespace(
        tp_size=2,
        ep_size=2,
        expert_tp_size=1,
        sp_size=2 if sequence_parallel else 1,
        dp_shard_size=2,
        model_parallel_enabled=True,
        device_mesh=mesh,
        device_type="cpu",
        dp_replicate_size=1,
        dp_enabled=True,
    )


def _tiny_dense_qwen_config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "attention_dropout": 0.0,
        "use_cache": False,
    }
    values.update(overrides)
    return Qwen3Config(**values)


def _engine_config():
    return OmegaConf.create(
        {
            "config": {"use_amp": False, "amp_dtype": "bf16"},
            "engine": {
                "cpu_offload": False,
                "activation_checkpoint": False,
                "checkpoint_every_n": 1,
                "find_unused_parameters": False,
                "overlap_backward_reductions": True,
            },
            "model": {"learning_rate": 1e-3, "clip_grad_norm": 0.0},
            "optim": {
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "weight_decay": 0.01,
            },
        }
    )


def test_qwen_plan_validates_head_and_expert_divisibility():
    with pytest.raises(ValueError, match="key/value heads"):
        build_qwen3_moe_plan(_tiny_qwen_config(num_key_value_heads=1), _pconfig())
    with pytest.raises(ValueError, match="experts"):
        build_qwen3_moe_plan(_tiny_qwen_config(num_experts=3), _pconfig())


def test_registry_builds_dense_and_moe_plans():
    dense = build_model_parallel_plan(_tiny_dense_qwen_config(), _dense_pconfig())
    moe = build_model_parallel_plan(_tiny_qwen_config(), _pconfig())
    assert dense.model_type == "qwen3"
    assert dense.expert_parallel_size == 1
    assert not dense.capabilities.expert_parallel
    assert moe.model_type == "qwen3_moe"
    assert moe.capabilities.expert_parallel


def test_expert_partition_localizes_replicated_routes():
    partition = ExpertPartition(num_experts=8, ep_size=2, ep_rank=1)
    indices = torch.tensor([[0, 6], [4, 3]])
    scores = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    local_indices, local_scores, mask = partition.localize_routes(indices, scores)
    torch.testing.assert_close(local_indices, torch.tensor([[4, 2], [0, 4]]))
    torch.testing.assert_close(local_scores, torch.tensor([[0.0, 0.8], [0.7, 0.0]]))
    torch.testing.assert_close(mask, torch.tensor([[False, True], [True, False]]))


def _run_qwen_folded_parallel_parity(
    rank, world_size, rendezvous_file, model_path, checkpoint_path, sequence_parallel
):
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh(
            "cpu",
            (1, world_size, 1),
            mesh_dim_names=("dp_replicate", "tp", "dp_shard"),
        )
        pconfig = _pconfig(
            mesh, world_size, sequence_parallel=sequence_parallel
        )
        config = Qwen3MoeConfig.from_pretrained(model_path)

        reference = Qwen3MoeForCausalLM.from_pretrained(model_path)
        reference.train()
        model, plan = load_causal_lm(
            OmegaConf.create({"model": {"name": str(model_path)}}), pconfig
        )
        assert model.config.output_router_logits
        model.train()
        engine = ParallelEngine(
            model,
            None,
            pconfig,
            _engine_config(),
            torch.device("cpu"),
            mp_plan=plan,
        )

        input_ids = torch.tensor([[1, 5, 7, 3], [2, 4, 6, 8]])
        labels = torch.tensor([[5, 7, 3, 9], [4, 6, 8, 10]])

        reference_outputs = reference(input_ids=input_ids)
        reference_logits = reference_outputs.logits
        reference_loss = F.cross_entropy(
            reference_logits.reshape(-1, config.vocab_size), labels.reshape(-1)
        )
        reference_loss = (
            reference_loss
            + reference.router_aux_loss_coef * reference_outputs.aux_loss
        )
        reference_loss.backward()

        loss = engine.forward(input_ids, labels).loss
        engine.backward(loss)
        torch.testing.assert_close(loss, reference_loss)

        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            reference_gradient = reference_parameters[name].grad
            assert parameter.grad is not None, name
            placement = plan.placement_for_parameter(name)
            if placement.shard_dim is None:
                expected = reference_gradient
            else:
                expected = reference_gradient.chunk(
                    world_size, dim=placement.shard_dim
                )[rank]
            torch.testing.assert_close(
                parameter.grad,
                expected,
                rtol=2e-2,
                atol=1e-4,
                msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
            )

            if name.endswith(".mlp.gate.weight"):
                gathered_router_gradients = [
                    torch.empty_like(parameter.grad) for _ in range(world_size)
                ]
                dist.all_gather(
                    gathered_router_gradients,
                    parameter.grad,
                    group=mesh.get_group("tp"),
                )
                for gathered_gradient in gathered_router_gradients[1:]:
                    torch.testing.assert_close(
                        gathered_gradient,
                        gathered_router_gradients[0],
                        rtol=0,
                        atol=0,
                    )

        reference_norm = torch.sqrt(
            sum(
                parameter.grad.detach().float().pow(2).sum()
                for parameter in reference.parameters()
                if parameter.grad is not None
            )
        )
        parallel_norm = engine.mp_wrapper.clip_grad_norm_(
            engine.optimizer_params, max_norm=1e9
        )
        torch.testing.assert_close(parallel_norm, reference_norm, rtol=2e-5, atol=1e-6)

        full_state = engine.state_dict()
        if rank == 0:
            reference_state = reference.state_dict()
            assert list(full_state) == list(reference_state)
            for name, expected in reference_state.items():
                torch.testing.assert_close(
                    full_state[name],
                    expected,
                    msg=lambda message, parameter_name=name: (
                        f"portable state {parameter_name}: {message}"
                    ),
                )

        full_state = engine.state_dict()
        if rank == 0:
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(full_state[name], expected)
        expected_local = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        for parameter in model.parameters():
            parameter.detach().zero_()
        engine.load_state_dict(full_state)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter,
                expected_local[name],
                msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
            )

        local_checkpoint = engine.sharded_state_dict()
        for parameter in model.parameters():
            parameter.detach().fill_(99)
        engine.load_sharded_state_dict(local_checkpoint)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, expected_local[name])

        engine.step()
        assert all(parameter.grad is None for parameter in model.parameters())
        stepped_local = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        engine.save_checkpoint(checkpoint_path, step=11)
        for parameter in model.parameters():
            parameter.detach().zero_()
        engine.optimizer.state.clear()
        resume = engine.load_checkpoint(checkpoint_path)
        assert resume["step"] == 11
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, stepped_local[name])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_qwen_folded_tensor_and_expert_parallel_matches_reference(tmp_path):
    torch.manual_seed(1234)
    model_path = tmp_path / "tiny-qwen"
    Qwen3MoeForCausalLM(_tiny_qwen_config()).save_pretrained(model_path)
    rendezvous_file = tmp_path / "rendezvous"
    mp.spawn(
        _run_qwen_folded_parallel_parity,
        args=(2, rendezvous_file, model_path, tmp_path / "checkpoint", False),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_qwen_sequence_and_expert_parallel_matches_reference(tmp_path):
    torch.manual_seed(1234)
    model_path = tmp_path / "tiny-qwen-sp"
    Qwen3MoeForCausalLM(_tiny_qwen_config()).save_pretrained(model_path)
    mp.spawn(
        _run_qwen_folded_parallel_parity,
        args=(
            2,
            tmp_path / "sp-rendezvous",
            model_path,
            tmp_path / "sp-checkpoint",
            True,
        ),
        nprocs=2,
        join=True,
    )


def _run_dense_qwen_parallel_parity(
    rank, world_size, rendezvous_file, model_path, sequence_parallel
):
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh(
            "cpu",
            (1, world_size, 1),
            mesh_dim_names=("dp_replicate", "tp", "dp_shard"),
        )
        pconfig = _dense_pconfig(
            mesh, world_size, sequence_parallel=sequence_parallel
        )
        config = Qwen3Config.from_pretrained(model_path)
        reference = Qwen3ForCausalLM.from_pretrained(model_path).train()
        model, plan = load_causal_lm(
            OmegaConf.create({"model": {"name": str(model_path)}}), pconfig
        )
        model.train()
        engine = ParallelEngine(
            model, None, pconfig, _engine_config(), torch.device("cpu"), mp_plan=plan
        )

        input_ids = torch.tensor([[1, 5, 7, 3], [2, 4, 6, 8]])
        labels = torch.tensor([[5, 7, 3, 9], [4, 6, 8, 10]])
        reference_loss = F.cross_entropy(
            reference(input_ids=input_ids).logits.reshape(-1, config.vocab_size),
            labels.reshape(-1),
        )
        reference_loss.backward()
        loss = engine.forward(input_ids, labels).loss
        engine.backward(loss)
        torch.testing.assert_close(loss, reference_loss)

        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            placement = plan.placement_for_parameter(name)
            expected = reference_parameters[name].grad
            if placement.shard_dim is not None:
                expected = expected.chunk(world_size, dim=placement.shard_dim)[rank]
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-2, atol=1e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_dense_qwen_tensor_parallel_matches_reference(tmp_path):
    torch.manual_seed(4321)
    model_path = tmp_path / "tiny-dense-qwen"
    Qwen3ForCausalLM(_tiny_dense_qwen_config()).save_pretrained(model_path)
    mp.spawn(
        _run_dense_qwen_parallel_parity,
        args=(2, tmp_path / "dense-rendezvous", model_path, False),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_dense_qwen_sequence_parallel_matches_reference(tmp_path):
    torch.manual_seed(4321)
    model_path = tmp_path / "tiny-dense-qwen-sp"
    Qwen3ForCausalLM(_tiny_dense_qwen_config()).save_pretrained(model_path)
    mp.spawn(
        _run_dense_qwen_parallel_parity,
        args=(2, tmp_path / "dense-sp-rendezvous", model_path, True),
        nprocs=2,
        join=True,
    )


def _run_composed_parallel_parity(
    rank,
    world_size,
    rendezvous_file,
    model_path,
    checkpoint_path,
    sequence_parallel,
):
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh(
            "cpu",
            (1, 2, 2),
            mesh_dim_names=("dp_replicate", "tp", "dp_shard"),
        )
        pconfig = _composed_pconfig(
            mesh, sequence_parallel=sequence_parallel
        )
        tp_rank = mesh.get_local_rank("tp")
        dp_shard_rank = mesh.get_local_rank("dp_shard")
        config = Qwen3MoeConfig.from_pretrained(model_path)

        reference = Qwen3MoeForCausalLM.from_pretrained(model_path).train()
        model, plan = load_causal_lm(
            OmegaConf.create({"model": {"name": str(model_path)}}), pconfig
        )
        model.train()
        engine = ParallelEngine(
            model, None, pconfig, _engine_config(), torch.device("cpu"), mp_plan=plan
        )

        input_ids = torch.tensor([[1, 5, 7, 3], [2, 4, 6, 8]]) + dp_shard_rank
        labels = torch.tensor([[5, 7, 3, 9], [4, 6, 8, 10]]) + dp_shard_rank
        reference_outputs = reference(input_ids=input_ids)
        reference_loss = F.cross_entropy(
            reference_outputs.logits.reshape(-1, config.vocab_size),
            labels.reshape(-1),
        )
        reference_loss = (
            reference_loss
            + reference.router_aux_loss_coef * reference_outputs.aux_loss
        )
        reference_loss.backward()

        loss = engine.forward(input_ids, labels).loss
        engine.backward(loss)
        torch.testing.assert_close(loss, reference_loss)

        dp_group = mesh.get_group("dp_shard")
        for parameter in reference.parameters():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=dp_group)
            parameter.grad.div_(2)

        reference_parameters = dict(reference.named_parameters())
        for unit in engine.fsdp_wrapper.units:
            for meta in unit.param_metas:
                expected = reference_parameters[meta.name].grad
                placement = plan.placement_for_parameter(meta.name)
                if placement.shard_dim is not None:
                    expected = expected.chunk(2, dim=placement.shard_dim)[tp_rank]
                start = min(dp_shard_rank * meta.padded_rows, expected.shape[0])
                expected = expected.narrow(0, start, meta.local_rows)
                actual = meta.sharded_param.grad._local_tensor
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-2,
                    atol=1e-4,
                    msg=lambda message, parameter_name=meta.name: (
                        f"{parameter_name}: {message}"
                    ),
                )

        reference_norm = torch.sqrt(
            sum(
                parameter.grad.detach().float().pow(2).sum()
                for parameter in reference.parameters()
            )
        )
        parallel_norm = engine.mp_wrapper.clip_grad_norm_(
            engine.optimizer_params, max_norm=1e9
        )
        torch.testing.assert_close(parallel_norm, reference_norm, rtol=2e-5, atol=1e-6)

        full_state = engine.state_dict()
        if rank == 0:
            reference_state = reference.state_dict()
            assert list(full_state) == list(reference_state)
            for name, expected in reference_state.items():
                torch.testing.assert_close(
                    full_state[name],
                    expected,
                    msg=lambda message, parameter_name=name: (
                        f"portable state {parameter_name}: {message}"
                    ),
                )

        local_parameters = [
            meta.sharded_param._local_tensor
            for unit in engine.fsdp_wrapper.units
            for meta in unit.param_metas
        ]
        expected_local = [parameter.detach().clone() for parameter in local_parameters]
        for parameter in local_parameters:
            parameter.zero_()
        engine.load_state_dict(full_state)
        for actual, expected in zip(local_parameters, expected_local, strict=True):
            torch.testing.assert_close(actual, expected)

        sharded_state = engine.sharded_state_dict()
        for parameter in local_parameters:
            parameter.fill_(rank + 1)
        engine.load_sharded_state_dict(sharded_state)
        for actual, expected in zip(local_parameters, expected_local, strict=True):
            torch.testing.assert_close(actual, expected)

        engine.step()
        expected_after_step = [
            parameter.detach().clone() for parameter in local_parameters
        ]
        expected_optimizer_tensors = [
            (
                value._local_tensor.detach().clone()
                if hasattr(value, "_local_tensor")
                else value.detach().clone()
            )
            for state in engine.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        ]
        engine.save_checkpoint(
            checkpoint_path,
            step=17,
            dataloader_state={"position": rank + 3},
            metadata={"test": "tp-ep-fsdp"},
        )
        for parameter in local_parameters:
            parameter.fill_(rank + 7)
        for state in engine.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
        restored = engine.load_checkpoint(checkpoint_path)
        assert restored["step"] == 17
        assert restored["dataloader_state"] == {"position": rank + 3}
        assert restored["metadata"] == {"test": "tp-ep-fsdp"}
        for actual, expected in zip(
            local_parameters, expected_after_step, strict=True
        ):
            torch.testing.assert_close(actual, expected)
        actual_optimizer_tensors = [
            value._local_tensor if hasattr(value, "_local_tensor") else value
            for state in engine.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        ]
        assert len(actual_optimizer_tensors) == len(expected_optimizer_tensors)
        for actual, expected in zip(
            actual_optimizer_tensors, expected_optimizer_tensors, strict=True
        ):
            torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_folded_tp_ep_composes_with_fsdp(tmp_path):
    torch.manual_seed(9876)
    model_path = tmp_path / "tiny-composed-qwen"
    Qwen3MoeForCausalLM(_tiny_qwen_config()).save_pretrained(model_path)
    mp.spawn(
        _run_composed_parallel_parity,
        args=(
            4,
            tmp_path / "composed-rendezvous",
            model_path,
            tmp_path / "composed-checkpoint",
            False,
        ),
        nprocs=4,
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_sequence_tp_ep_composes_with_fsdp(tmp_path):
    torch.manual_seed(9876)
    model_path = tmp_path / "tiny-composed-sp-qwen"
    Qwen3MoeForCausalLM(_tiny_qwen_config()).save_pretrained(model_path)
    mp.spawn(
        _run_composed_parallel_parity,
        args=(
            4,
            tmp_path / "composed-sp-rendezvous",
            model_path,
            tmp_path / "composed-sp-checkpoint",
            True,
        ),
        nprocs=4,
        join=True,
    )

"""Opt-in multi-GPU integration tests for the distributed engine.

Run the full suite on a node with four visible GPUs using:

    uv run pytest -q -m gpu tests/gpu

Each test owns the first N visible CUDA devices and launches its own NCCL
process group. These tests intentionally use tiny local Qwen checkpoints, so
they validate distributed semantics without downloading production weights.
"""

from __future__ import annotations

import os
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.distributed.tensor import DTensor
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from parallel.parallel.engine.dp_sharded import FSDPWrapper
from parallel.parallel.engine.engine import ParallelEngine
from parallel.parallel.engine.model_loading import load_causal_lm
from parallel.parallel.state import ParallelConfig, Strategies


pytestmark = pytest.mark.gpu


def _requires_gpus(count: int):
    available = dist.is_available() and dist.is_nccl_available()
    available = available and torch.cuda.device_count() >= count
    return pytest.mark.skipif(
        not available,
        reason=f"requires NCCL and at least {count} visible CUDA devices",
    )


def _tiny_dense_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        use_cache=False,
    )


def _tiny_moe_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
        output_router_logits=True,
        router_aux_loss_coef=0.01,
        use_cache=False,
    )


def _save_bf16_model(model: nn.Module, path: Path) -> None:
    model.to(dtype=torch.bfloat16)
    model.save_pretrained(path)


def _runtime_config(
    model_path: Path,
    *,
    tp: int,
    ep: int,
    dp_shard: int,
    sp: int = 1,
    overlap_backward_reductions: bool = True,
):
    return OmegaConf.create(
        {
            "model": {
                "name": str(model_path),
                "learning_rate": 1e-3,
                "clip_grad_norm": 0.0,
                "output_router_logits": True,
            },
            "optim": {
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "weight_decay": 0.01,
            },
            "config": {
                "device_type": "cuda",
                "backend": "nccl",
                "use_amp": False,
                "amp_dtype": "bf16",
            },
            "engine": {
                "cpu_offload": False,
                "activation_checkpoint": False,
                "checkpoint_every_n": 1,
                "find_unused_parameters": False,
                "overlap_backward_reductions": overlap_backward_reductions,
            },
            "parallel": {
                "dp_replicate": 1,
                "dp_shard": dp_shard,
                "tp": tp,
                "cp": 1,
                "sp": sp,
                "ep": ep,
                "expert_tp": 1,
                "pp": 1,
            },
        }
    )


def _init_nccl(rank: int, world_size: int, rendezvous_file: Path) -> torch.device:
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=3),
    )
    return torch.device("cuda", rank)


def _progress(rank: int, stage: str) -> None:
    print(f"[GPU integration rank {rank}] {stage}", flush=True)


def _parallel_config(cfg) -> ParallelConfig:
    pconfig = ParallelConfig(cfg)
    pconfig.set_device_mesh("cuda")
    return pconfig


def _reference_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )


def _moe_loss(model, input_ids, labels):
    outputs = model(input_ids=input_ids)
    token_loss = F.cross_entropy(
        outputs.logits.reshape(-1, outputs.logits.shape[-1]),
        labels.reshape(-1),
    )
    loss = token_loss + model.router_aux_loss_coef * outputs.aux_loss
    return outputs, loss


def _average_reference_gradients(model, group, divisor: int) -> None:
    for parameter in model.parameters():
        assert parameter.grad is not None
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
        parameter.grad.div_(divisor)


def _expected_local_shard(reference_parameter, meta, placement, tp_rank, dp_rank):
    expected = reference_parameter
    if placement.shard_dim is not None:
        expected = expected.chunk(2, dim=placement.shard_dim)[tp_rank]
    start = min(dp_rank * meta.padded_rows, expected.shape[0])
    return expected.narrow(0, start, meta.local_rows)


def _assert_composed_parameters(engine, reference, plan, pconfig) -> None:
    reference_parameters = dict(reference.named_parameters())
    tp_rank = pconfig.device_mesh.get_local_rank(Strategies.TP)
    dp_rank = pconfig.device_mesh.get_local_rank(Strategies.DP_SHARD)
    for unit in engine.fsdp_wrapper.units:
        for meta in unit.param_metas:
            placement = plan.placement_for_parameter(meta.name)
            expected = _expected_local_shard(
                reference_parameters[meta.name].detach(),
                meta,
                placement,
                tp_rank,
                dp_rank,
            )
            torch.testing.assert_close(
                meta.sharded_param._local_tensor,
                expected,
                rtol=5e-2,
                atol=5e-3,
                msg=lambda message, name=meta.name: f"{name}: {message}",
            )


def _assert_composed_gradients(engine, reference, plan, pconfig) -> None:
    reference_parameters = dict(reference.named_parameters())
    tp_rank = pconfig.device_mesh.get_local_rank(Strategies.TP)
    dp_rank = pconfig.device_mesh.get_local_rank(Strategies.DP_SHARD)
    for unit in engine.fsdp_wrapper.units:
        for meta in unit.param_metas:
            placement = plan.placement_for_parameter(meta.name)
            expected = _expected_local_shard(
                reference_parameters[meta.name].grad,
                meta,
                placement,
                tp_rank,
                dp_rank,
            )
            actual = meta.sharded_param.grad._local_tensor
            torch.testing.assert_close(
                actual,
                expected,
                rtol=7e-2,
                atol=7e-3,
                msg=lambda message, name=meta.name: f"{name}: {message}",
            )


class _TinyFSDPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(16, 32, bias=False)
        self.output = nn.Linear(32, 8, bias=False)

    def forward(self, inputs):
        return self.output(F.gelu(self.input(inputs)))


def _run_fsdp_gpu(rank, world_size, rendezvous_file):
    device = _init_nccl(rank, world_size, rendezvous_file)
    try:
        torch.manual_seed(1234)
        model = _TinyFSDPModel().to(device=device, dtype=torch.bfloat16)
        reference = _TinyFSDPModel().to(device=device, dtype=torch.bfloat16)
        reference.load_state_dict(model.state_dict())
        cfg = _runtime_config(Path("unused"), tp=1, ep=1, dp_shard=2)
        pconfig = _parallel_config(cfg)
        wrapper = FSDPWrapper(model, pconfig, device=device)
        wrapper.shard_model()
        optimizer = torch.optim.SGD(wrapper.get_optimizer_params(), lr=0.05)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)

        inputs = (
            torch.arange(64, device=device, dtype=torch.float32).reshape(4, 16)
            .add_(rank * 64)
            .div_(100)
            .to(torch.bfloat16)
        )
        loss = model(inputs).float().square().mean()
        wrapper.prepare_backward(loss)
        loss.backward()
        wrapper.finalize_backward()

        reference_loss = reference(inputs).float().square().mean()
        reference_loss.backward()
        _average_reference_gradients(reference, dist.group.WORLD, world_size)
        optimizer.step()
        reference_optimizer.step()

        state = wrapper.full_state_dict()
        if rank == 0:
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    state[name].to(device), expected, rtol=5e-2, atol=5e-3
                )
    finally:
        dist.destroy_process_group()


@_requires_gpus(2)
def test_fsdp_two_gpus(tmp_path):
    mp.spawn(
        _run_fsdp_gpu,
        args=(2, tmp_path / "fsdp-nccl-rendezvous"),
        nprocs=2,
        join=True,
    )


def _run_dense_tp_gpu(
    rank, world_size, rendezvous_file, model_path, sequence_parallel=False
):
    device = _init_nccl(rank, world_size, rendezvous_file)
    try:
        cfg = _runtime_config(
            model_path,
            tp=2,
            ep=1,
            dp_shard=1,
            sp=2 if sequence_parallel else 1,
        )
        pconfig = _parallel_config(cfg)
        reference = Qwen3ForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to(device)
        reference.train()
        model, plan = load_causal_lm(cfg, pconfig)
        model.train()
        engine = ParallelEngine(model, None, pconfig, cfg, device, mp_plan=plan)

        input_ids = torch.tensor(
            [[1, 5, 7, 3], [2, 4, 6, 8]], device=device
        )
        labels = torch.tensor([[5, 7, 3, 9], [4, 6, 8, 10]], device=device)
        reference_loss = F.cross_entropy(
            reference(input_ids=input_ids).logits.float().reshape(-1, 32),
            labels.reshape(-1),
        )
        reference_loss.backward()
        outputs = engine.forward(input_ids, labels)
        engine.backward(outputs.loss)
        torch.testing.assert_close(
            outputs.loss, reference_loss, rtol=5e-2, atol=5e-3
        )

        reference_parameters = dict(reference.named_parameters())
        tp_rank = pconfig.device_mesh.get_local_rank(Strategies.TP)
        for name, parameter in model.named_parameters():
            placement = plan.placement_for_parameter(name)
            expected = reference_parameters[name].grad
            if placement.shard_dim is not None:
                expected = expected.chunk(2, dim=placement.shard_dim)[tp_rank]
            torch.testing.assert_close(
                parameter.grad,
                expected,
                rtol=7e-2,
                atol=7e-3,
                msg=lambda message, parameter_name=name: (
                    f"{parameter_name}: {message}"
                ),
            )
    finally:
        dist.destroy_process_group()


@_requires_gpus(2)
def test_dense_tensor_parallel_two_gpus(tmp_path):
    model_path = tmp_path / "tiny-dense-qwen-bf16"
    _save_bf16_model(Qwen3ForCausalLM(_tiny_dense_config()), model_path)
    mp.spawn(
        _run_dense_tp_gpu,
        args=(2, tmp_path / "dense-tp-nccl-rendezvous", model_path),
        nprocs=2,
        join=True,
    )


@_requires_gpus(2)
def test_dense_tensor_and_sequence_parallel_two_gpus(tmp_path):
    model_path = tmp_path / "tiny-dense-qwen-sp-bf16"
    _save_bf16_model(Qwen3ForCausalLM(_tiny_dense_config()), model_path)
    mp.spawn(
        _run_dense_tp_gpu,
        args=(2, tmp_path / "dense-sp-nccl-rendezvous", model_path, True),
        nprocs=2,
        join=True,
    )


def _run_folded_tp_ep_gpu(
    rank, world_size, rendezvous_file, model_path, sequence_parallel=False
):
    device = _init_nccl(rank, world_size, rendezvous_file)
    try:
        cfg = _runtime_config(
            model_path,
            tp=2,
            ep=2,
            dp_shard=1,
            sp=2 if sequence_parallel else 1,
        )
        pconfig = _parallel_config(cfg)
        reference = Qwen3MoeForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to(device)
        reference.train()
        model, plan = load_causal_lm(cfg, pconfig)
        model.train()
        engine = ParallelEngine(model, None, pconfig, cfg, device, mp_plan=plan)

        input_ids = torch.tensor(
            [[1, 5, 7, 3], [2, 4, 6, 8]], device=device
        )
        labels = torch.tensor([[5, 7, 3, 9], [4, 6, 8, 10]], device=device)
        reference_outputs, reference_loss = _moe_loss(reference, input_ids, labels)
        reference_loss.backward()
        outputs = engine.forward(input_ids, labels)
        assert outputs.aux_loss is not None
        engine.backward(outputs.loss)
        torch.testing.assert_close(
            outputs.aux_loss,
            reference_outputs.aux_loss,
            rtol=5e-2,
            atol=5e-3,
        )
        torch.testing.assert_close(
            outputs.loss, reference_loss, rtol=5e-2, atol=5e-3
        )

        reference_parameters = dict(reference.named_parameters())
        tp_rank = pconfig.device_mesh.get_local_rank(Strategies.TP)
        for name, parameter in model.named_parameters():
            placement = plan.placement_for_parameter(name)
            expected = reference_parameters[name].grad
            if placement.shard_dim is not None:
                expected = expected.chunk(2, dim=placement.shard_dim)[tp_rank]
            torch.testing.assert_close(
                parameter.grad, expected, rtol=7e-2, atol=7e-3
            )
    finally:
        dist.destroy_process_group()


@_requires_gpus(2)
def test_folded_tensor_and_expert_parallel_two_gpus(tmp_path):
    model_path = tmp_path / "tiny-moe-qwen-bf16"
    _save_bf16_model(Qwen3MoeForCausalLM(_tiny_moe_config()), model_path)
    mp.spawn(
        _run_folded_tp_ep_gpu,
        args=(2, tmp_path / "folded-nccl-rendezvous", model_path),
        nprocs=2,
        join=True,
    )


@_requires_gpus(2)
def test_sequence_tensor_and_expert_parallel_two_gpus(tmp_path):
    model_path = tmp_path / "tiny-moe-qwen-sp-bf16"
    _save_bf16_model(Qwen3MoeForCausalLM(_tiny_moe_config()), model_path)
    mp.spawn(
        _run_folded_tp_ep_gpu,
        args=(2, tmp_path / "folded-sp-nccl-rendezvous", model_path, True),
        nprocs=2,
        join=True,
    )


def _composed_step(
    engine, reference, reference_optimizer, plan, pconfig, offset, rank
):
    device = pconfig.device
    data_rank = pconfig.data_rank
    input_ids = (
        torch.tensor([[1, 5, 7, 3], [2, 4, 6, 8]], device=device)
        + data_rank
        + offset
    )
    labels = (
        torch.tensor([[5, 7, 3, 9], [4, 6, 8, 10]], device=device)
        + data_rank
        + offset
    )

    _progress(rank, f"step {offset + 1}: reference forward")
    reference_outputs, reference_loss = _moe_loss(reference, input_ids, labels)
    reference_loss.backward()
    _progress(rank, f"step {offset + 1}: parallel forward")
    outputs = engine.forward(input_ids, labels)
    assert outputs.aux_loss is not None
    _progress(rank, f"step {offset + 1}: parallel backward")
    engine.backward(outputs.loss)
    _progress(rank, f"step {offset + 1}: backward complete")

    torch.testing.assert_close(
        outputs.aux_loss,
        reference_outputs.aux_loss,
        rtol=5e-2,
        atol=5e-3,
    )
    torch.testing.assert_close(
        outputs.loss, reference_loss, rtol=5e-2, atol=5e-3
    )

    dp_group = pconfig.device_mesh.get_group(Strategies.DP_SHARD)
    _average_reference_gradients(reference, dp_group, pconfig.dp_shard_size)
    _progress(rank, f"step {offset + 1}: reference gradients reduced")
    _assert_composed_gradients(engine, reference, plan, pconfig)

    reference_norm = torch.sqrt(
        sum(
            parameter.grad.detach().float().pow(2).sum()
            for parameter in reference.parameters()
        )
    )
    parallel_norm = engine.mp_wrapper.clip_grad_norm_(
        engine.optimizer_params, max_norm=1e9
    )
    torch.testing.assert_close(
        parallel_norm, reference_norm, rtol=2e-2, atol=2e-3
    )
    _progress(rank, f"step {offset + 1}: gradient norm checked")

    engine.step()
    reference_optimizer.step()
    reference_optimizer.zero_grad(set_to_none=True)
    _assert_composed_parameters(engine, reference, plan, pconfig)
    _progress(rank, f"step {offset + 1}: optimizer step checked")
    return outputs.loss.detach()


def _run_composed_gpu(
    rank,
    world_size,
    rendezvous_file,
    model_path,
    checkpoint_path,
    overlap_backward_reductions,
    sequence_parallel=False,
):
    device = _init_nccl(rank, world_size, rendezvous_file)
    try:
        _progress(rank, "NCCL initialized")
        cfg = _runtime_config(
            model_path,
            tp=2,
            ep=2,
            dp_shard=2,
            sp=2 if sequence_parallel else 1,
            overlap_backward_reductions=overlap_backward_reductions,
        )
        pconfig = _parallel_config(cfg)
        reference = Qwen3MoeForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to(device)
        reference.train()
        reference_optimizer = _reference_optimizer(reference)
        _progress(rank, "reference model loaded")

        model, plan = load_causal_lm(cfg, pconfig)
        model.train()
        engine = ParallelEngine(model, None, pconfig, cfg, device, mp_plan=plan)
        _progress(rank, "parallel engine initialized")
        _composed_step(
            engine,
            reference,
            reference_optimizer,
            plan,
            pconfig,
            offset=0,
            rank=rank,
        )

        _progress(rank, "portable state gather")
        portable_state = engine.state_dict()
        if rank == 0:
            reference_state = reference.state_dict()
            assert list(portable_state) == list(reference_state)
            for name, expected in reference_state.items():
                torch.testing.assert_close(
                    portable_state[name].to(device),
                    expected,
                    rtol=5e-2,
                    atol=5e-3,
                )

        expected_local = [
            (meta.name, meta.sharded_param._local_tensor.detach().clone())
            for unit in engine.fsdp_wrapper.units
            for meta in unit.param_metas
        ]
        for parameter in engine.fsdp_wrapper.get_sharded_params():
            parameter._local_tensor.zero_()
        _progress(rank, "portable state load")
        engine.load_state_dict(portable_state)
        _progress(rank, "portable state load returned")
        actual_local = [
            (meta.name, meta.sharded_param._local_tensor)
            for unit in engine.fsdp_wrapper.units
            for meta in unit.param_metas
        ]
        for (actual_name, actual), (expected_name, expected) in zip(
            actual_local, expected_local, strict=True
        ):
            assert actual_name == expected_name
            torch.testing.assert_close(
                actual,
                expected,
                msg=lambda message, name=actual_name: (
                    f"portable restore mismatch for {name}: {message}"
                ),
            )
        _progress(rank, "portable state restored")

        _progress(rank, "exact checkpoint save")
        engine.save_checkpoint(
            checkpoint_path,
            step=1,
            dataloader_state={"position": rank + 10},
            metadata={
                "overlap_backward_reductions": overlap_backward_reductions
            },
        )
        _progress(rank, "exact checkpoint saved")

        del engine, model
        torch.cuda.empty_cache()
        _progress(rank, "resume model load")
        resumed_model, resumed_plan = load_causal_lm(cfg, pconfig)
        resumed_model.train()
        resumed = ParallelEngine(
            resumed_model,
            None,
            pconfig,
            cfg,
            device,
            mp_plan=resumed_plan,
        )
        _progress(rank, "resume engine initialized")
        resume_state = resumed.load_checkpoint(checkpoint_path)
        assert resume_state["step"] == 1
        assert resume_state["dataloader_state"] == {"position": rank + 10}
        assert resume_state["metadata"] == {
            "overlap_backward_reductions": overlap_backward_reductions
        }
        _assert_composed_parameters(resumed, reference, resumed_plan, pconfig)
        _progress(rank, "exact checkpoint restored")

        _composed_step(
            resumed,
            reference,
            reference_optimizer,
            resumed_plan,
            pconfig,
            offset=1,
            rank=rank,
        )
        _progress(rank, "composed test complete")
    except BaseException:
        print(
            f"[GPU integration rank {rank}] ORIGINAL WORKER EXCEPTION",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("overlap_backward_reductions", [True, False])
@_requires_gpus(4)
def test_folded_tp_ep_composes_with_fsdp_four_gpus(
    tmp_path, overlap_backward_reductions
):
    mode = "overlap" if overlap_backward_reductions else "deferred"
    model_path = tmp_path / f"tiny-composed-qwen-bf16-{mode}"
    _save_bf16_model(Qwen3MoeForCausalLM(_tiny_moe_config()), model_path)
    mp.spawn(
        _run_composed_gpu,
        args=(
            4,
            tmp_path / f"composed-nccl-rendezvous-{mode}",
            model_path,
            tmp_path / f"composed-checkpoint-{mode}",
            overlap_backward_reductions,
        ),
        nprocs=4,
        join=True,
    )


@pytest.mark.parametrize("overlap_backward_reductions", [True, False])
@_requires_gpus(4)
def test_sequence_tp_ep_composes_with_fsdp_four_gpus(
    tmp_path, overlap_backward_reductions
):
    mode = "overlap" if overlap_backward_reductions else "deferred"
    model_path = tmp_path / f"tiny-composed-sp-qwen-bf16-{mode}"
    _save_bf16_model(Qwen3MoeForCausalLM(_tiny_moe_config()), model_path)
    mp.spawn(
        _run_composed_gpu,
        args=(
            4,
            tmp_path / f"composed-sp-nccl-rendezvous-{mode}",
            model_path,
            tmp_path / f"composed-sp-checkpoint-{mode}",
            overlap_backward_reductions,
            True,
        ),
        nprocs=4,
        join=True,
    )

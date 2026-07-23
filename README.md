# parallel

`parallel` is a small, pedagogical, explicit distributed-training engine for studying how
PyTorch model parallelism, expert parallelism, sequence parallelism, and fully
sharded data parallelism fit together.

The main supported model families are dense Qwen3 and Qwen3-MoE. The engine
implements:

- Replicated data parallelism (`DP replicate`).
- Fully sharded data parallelism (`FSDP` / `DP shard`).
- Tensor parallel attention, MLPs, embeddings, and LM heads (`TP`).
- Sequence-parallel decoder activations (`SP`).
- Expert ownership and variable-size token all-to-all (`EP`).
- Vocabulary-parallel cross entropy.
- MoE router auxiliary loss.
- Distributed gradient norm and clipping.
- Portable full-model and exact-topology training checkpoints.
- Aggregate collective profiling and PyTorch timeline traces.

The implementation deliberately keeps the distributed operations visible. It
is intended as both a training system and a place to understand the mechanics
of composable parallelism.

## Setup

## Download the training data

```bash
uv run python -m parallel.parallel.dataset -n 50 -w 24
```

## Parallel topology

The currently supported physical world size is:

```text
world_size = dp_replicate × dp_shard × tp
```

`EP` and `SP` are folded onto the TP ranks and therefore do not multiply the
world size:

```text
MoE:    ep == tp
SP:     sp == 1 or sp == tp
ETP:    expert_tp == 1
```

For example:

```text
4 GPUs
├── TP = EP = SP = 2
└── DP shard = 2
```

Each pair of model-parallel ranks owns one TP/EP/SP model replica. The
corresponding parameters of that model-parallel replica are then sharded over
a two-rank FSDP group.

### What each axis does

- `parallel.dp_replicate`: replicas hold the same parameters and average
  gradients.
- `parallel.dp_shard`: FSDP shards parameters, gradients, and optimizer state;
  parameters are all-gathered by transformer unit for computation.
- `parallel.tp`: shards attention heads, MLP dimensions, embeddings, and the
  vocabulary projection over one model-parallel group.
- `parallel.sp`: keeps residual activations token-sharded between transformer
  operations. Attention/MLP inputs are gathered when required and row-parallel
  outputs are reduce-scattered.
- `parallel.ep`: assigns experts to the folded TP/EP ranks and dispatches routed
  tokens using variable-size all-to-all.
- `parallel.expert_tp`: reserved for tensor parallelism within an individual
  expert. It currently must be `1`.

Attention heads, key/value heads, sharded MLP dimensions, experts, and
vocabulary dimensions must be divisible by their relevant parallel sizes.

## Training

Commands below use offline W&B logging by default. Only rank zero creates a
run.

### Two-GPU FSDP

This keeps model parallelism disabled and shards the complete model over two
GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
WANDB_MODE=offline \
WANDB_API_KEY=offline \
uv run torchrun --standalone --nproc-per-node=2 \
  -m parallel.parallel.train \
  model.name=Qwen/Qwen3-0.6B \
  parallel.dp_replicate=1 \
  parallel.dp_shard=2 \
  parallel.tp=1 \
  parallel.sp=1 \
  parallel.ep=1 \
  model.per_device_batch_size=1 \
  model.max_seq_length=512 \
  config.grad_accum_steps=1 \
  engine.activation_checkpoint=true
```

### Two-GPU dense TP

Dense Qwen3 uses TP without EP:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
WANDB_MODE=offline \
WANDB_API_KEY=offline \
uv run torchrun --standalone --nproc-per-node=2 \
  -m parallel.parallel.train \
  model.name=Qwen/Qwen3-0.6B \
  parallel.dp_replicate=1 \
  parallel.dp_shard=1 \
  parallel.tp=2 \
  parallel.sp=1 \
  parallel.ep=1 \
  model.per_device_batch_size=1 \
  model.max_seq_length=512
```

Enable sequence parallelism on the same two TP ranks with:

```bash
parallel.sp=2
```

SP does not require two additional ranks.

### Four-GPU Qwen3-MoE with TP + SP + EP + FSDP

This is the primary composed configuration used during development:

```bash
set -o pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_MODE=offline \
WANDB_API_KEY=offline \
uv run torchrun --standalone --nproc-per-node=4 \
  -m parallel.parallel.train \
  parallel.dp_replicate=1 \
  parallel.dp_shard=2 \
  parallel.tp=2 \
  parallel.sp=2 \
  parallel.ep=2 \
  parallel.expert_tp=1 \
  model.per_device_batch_size=1 \
  model.max_seq_length=512 \
  config.grad_accum_steps=1 \
  config.max_steps=15 \
  config.eval_interval=100 \
  config.eval_steps=10 \
  config.checkpoint_interval=0 \
  engine.activation_checkpoint=true \
  engine.checkpoint_every_n=1 \
  engine.overlap_backward_reductions=true \
  engine.collective_profiling=false \
  engine.torch_profiler=false \
  2>&1 | tee /tmp/qwen3-training.log
```

## Project layout

```text
parallel/parallel/
├── train.py                         training loop
├── dataset.py                       dataset shard downloader/reader
├── state.py                         topology and process-group state
├── profiling.py                     aggregate and timeline profiling
└── engine/
    ├── engine.py                    forward/backward/optimizer orchestration
    ├── dp_sharded.py                FSDP wrapper and reduction schedule
    ├── _dp_param_unit.py            flat FSDP parameter/gradient units
    ├── checkpoint.py                training checkpoint manager
    └── model_parallel/
        ├── api.py                   model-parallel orchestration
        ├── plan.py                  model-independent placement plan
        ├── registry.py              model-family plan registry
        ├── tensor_parallel.py       TP operations
        ├── sequence_parallel.py     token-sharded residual layouts
        ├── expert_parallel.py       expert ownership
        ├── token_dispatch.py        variable-size EP all-to-all
        ├── loss_parallel.py         vocabulary-parallel loss
        ├── grad_norm.py             distributed gradient norm
        └── models/                  Qwen3/Qwen3-MoE plan builders
```

## Current constraints

- TP plans currently target Qwen3 and Qwen3-MoE.
- MoE uses folded `tp == ep`.
- Sequence parallelism is disabled or folded as `sp == tp`.
- Expert tensor parallelism currently requires `expert_tp == 1`.
- Pipeline and context-parallel axes are reserved but not implemented.
- `torch.compile` is not supported with the custom FSDP or folded
  model-parallel implementation.
- Multi-node untested

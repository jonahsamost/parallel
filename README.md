### Roadmap
- DP / FSDP / TP / EP / PP / CP axes
- activation checkpointing
- comms axes (i.e. DeepEP)

### Running

Launch replicated data parallel training with:

```bash
torchrun --nproc-per-node=8 -m parallel.parallel.train parallel.dp_replicate=8
```

Launch fully sharded data parallel training with:

```bash
torchrun --nproc-per-node=8 -m parallel.parallel.train parallel.dp_shard=8
```

With no model-parallel axis enabled, the product of `parallel.dp_replicate` and
`parallel.dp_shard` must equal the distributed world size.

Dense Qwen3 models support tensor parallelism:

```bash
torchrun --nproc-per-node=4 -m parallel.parallel.train parallel.tp=4
```

Qwen3-MoE models use a folded model-parallel group: the same ranks shard
attention heads with tensor parallelism and experts with expert parallelism:

```bash
torchrun --nproc-per-node=4 -m parallel.parallel.train \
  parallel.tp=4 parallel.ep=4 parallel.expert_tp=1
```

Folded MoE requires `tp == ep`, and expert tensor parallelism currently requires
`expert_tp == 1`. Attention heads, key/value heads, sharded MLP dimensions,
experts, and vocabulary size must be divisible by their parallel dimension.
Parameters are initialized on meta and only each rank's local model-parallel
checkpoint shards are materialized.

Tensor/expert parallelism composes with the custom FSDP implementation. For
example, four ranks can form two-rank folded TP/EP groups and independently
shard each local model-parallel parameter over a two-rank DP-shard group:

```bash
torchrun --nproc-per-node=4 -m parallel.parallel.train \
  parallel.tp=2 parallel.ep=2 parallel.expert_tp=1 \
  parallel.dp_shard=2
```

The physical world size is `dp_replicate * tp * dp_shard` (`ep` reuses the TP
axis). Portable full-model checkpoints reconstruct both shard axes, while
exact-topology training checkpoints retain TP and DP-shard ownership plus
optimizer and resume state. Sequence parallelism, expert tensor parallelism,
and model-parallel `torch.compile` remain follow-up work.

The custom FSDP implementation supports static module graphs, gradient
accumulation, separately sharded trainable and frozen parameters, mixed
BF16/FP32 parameter groups, CPU shard offload, activation checkpointing, and
full model state dictionaries. `torch.compile` is currently supported only
when `parallel.dp_shard=1`.

### Profiling

`profile` can time either a block or a decorated function:

```python
from parallel.parallel import profile

with profile("shard_to_device", synchronize=True, device=device):
    shard_on_device = shard_on_device.to(device, non_blocking=True)

@profile("forward")
def forward(inputs):
    return model(inputs)

with profile("all_gather", device=device, use_cuda_events=True):
    dist.all_gather_into_tensor(full_buf, shard, group=group)

with profile("training_step", device=device, use_torch_profiler=True):
    loss = model(inputs).loss
    loss.backward()
```

Timings are logged in milliseconds through the rank-aware project logger. Use
`synchronize=True` for accurate CUDA timings; it deliberately prevents CUDA
work from overlapping across the measured boundary. CUDA events measure work
on the current CUDA stream and wait for the end event when the block exits. The
PyTorch profiler logs its ten most expensive operators and is best enabled only
for a small number of iterations.

### Would be cool
- quantization aware training
- 8bit and [4bit](https://github.com/sgl-project/sglang/pull/26083)

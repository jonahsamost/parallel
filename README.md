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

For hybrid sharded data parallelism, the product of `parallel.dp_replicate` and
`parallel.dp_shard` must equal the distributed world size. Tensor, context,
sequence, expert, and pipeline parallel sizes currently must remain `1`.

The custom FSDP implementation supports static module graphs, gradient
accumulation, separately sharded trainable and frozen parameters, mixed
BF16/FP32 parameter groups, CPU shard offload, activation checkpointing, and
full model state dictionaries. `torch.compile` is currently supported only
when `parallel.dp_shard=1`.

### Would be cool
- quantization aware training
- 8bit and [4bit](https://github.com/sgl-project/sglang/pull/26083)

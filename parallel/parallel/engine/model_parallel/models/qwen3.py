from __future__ import annotations

from ..plan import (
    ModelParallelCapabilities,
    ModelParallelPlan,
    ModuleRule,
    ParameterParallelism,
    ParameterPlacement,
)
from ..registry import register_model_parallel_plan
from .common import (
    qwen_attention_placements,
    qwen_replicated_placements,
    require_divisible,
)


@register_model_parallel_plan("qwen3")
def build_qwen3_plan(config, pconfig) -> ModelParallelPlan:
    tp_size = pconfig.tp_size
    sp_size = getattr(pconfig, "sp_size", 1)
    if tp_size <= 1:
        raise ValueError("Qwen3 tensor parallelism requires tp > 1")
    if pconfig.ep_size != 1:
        raise ValueError("Dense Qwen3 requires ep == 1")
    if pconfig.expert_tp_size != 1:
        raise ValueError("Dense Qwen3 requires expert_tp == 1")
    if sp_size not in (1, tp_size):
        raise ValueError("Qwen3 sequence parallelism requires sp == 1 or sp == tp")
    sequence_parallel = sp_size > 1

    require_divisible(
        "Qwen3", "attention heads", config.num_attention_heads, tp_size
    )
    require_divisible("Qwen3", "key/value heads", config.num_key_value_heads, tp_size)
    require_divisible("Qwen3", "intermediate size", config.intermediate_size, tp_size)
    require_divisible("Qwen3", "vocabulary size", config.vocab_size, tp_size)

    rules = (
        ModuleRule("model.embed_tokens", "embedding_rowwise"),
        ModuleRule(
            "model.layers.*.self_attn.q_proj",
            "colwise_sequence" if sequence_parallel else "colwise",
        ),
        ModuleRule(
            "model.layers.*.self_attn.k_proj",
            "colwise_sequence" if sequence_parallel else "colwise",
        ),
        ModuleRule(
            "model.layers.*.self_attn.v_proj",
            "colwise_sequence" if sequence_parallel else "colwise",
        ),
        ModuleRule(
            "model.layers.*.self_attn.o_proj",
            "rowwise_sequence" if sequence_parallel else "rowwise",
        ),
        ModuleRule(
            "model.layers.*.mlp.gate_proj",
            "colwise_sequence" if sequence_parallel else "colwise",
        ),
        ModuleRule(
            "model.layers.*.mlp.up_proj",
            "colwise_sequence" if sequence_parallel else "colwise",
        ),
        ModuleRule(
            "model.layers.*.mlp.down_proj",
            "rowwise_sequence" if sequence_parallel else "rowwise",
        ),
        ModuleRule("lm_head", "colwise_sequence" if sequence_parallel else "colwise"),
    )
    P = ParameterParallelism
    placements = (
        qwen_attention_placements()
        + (
            ParameterPlacement("model.layers.*.mlp.gate_proj.weight", P.COLUMN, 0),
            ParameterPlacement("model.layers.*.mlp.up_proj.weight", P.COLUMN, 0),
            ParameterPlacement("model.layers.*.mlp.down_proj.weight", P.ROW, 1),
        )
        + qwen_replicated_placements(sequence_parallel)
    )
    return ModelParallelPlan(
        model_type=config.model_type,
        tensor_parallel_size=tp_size,
        expert_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel_size=sp_size,
        module_rules=rules,
        parameter_placements=placements,
        capabilities=ModelParallelCapabilities(
            expert_parallel=False,
            sequence_parallel=sequence_parallel,
            fsdp_composition=True,
        ),
    )

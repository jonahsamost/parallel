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


@register_model_parallel_plan("qwen3_moe")
def build_qwen3_moe_plan(config, pconfig) -> ModelParallelPlan:
    tp_size = pconfig.tp_size
    ep_size = pconfig.ep_size
    expert_tp_size = pconfig.expert_tp_size
    if tp_size <= 1 or ep_size <= 1:
        raise ValueError(
            "Qwen3-MoE folded model parallelism requires tp > 1 and ep > 1"
        )
    if tp_size != ep_size:
        raise ValueError("Qwen3-MoE folded model parallelism requires tp == ep")
    if expert_tp_size != 1:
        raise NotImplementedError("Qwen3-MoE expert tensor parallelism must be 1")

    require_divisible(
        "Qwen3-MoE", "attention heads", config.num_attention_heads, tp_size
    )
    require_divisible(
        "Qwen3-MoE", "key/value heads", config.num_key_value_heads, tp_size
    )
    require_divisible("Qwen3-MoE", "experts", config.num_experts, ep_size)
    require_divisible("Qwen3-MoE", "vocabulary size", config.vocab_size, tp_size)

    rules = (
        ModuleRule("model.embed_tokens", "embedding_rowwise"),
        ModuleRule("model.layers.*.self_attn.q_proj", "colwise"),
        ModuleRule("model.layers.*.self_attn.k_proj", "colwise"),
        ModuleRule("model.layers.*.self_attn.v_proj", "colwise"),
        ModuleRule("model.layers.*.self_attn.o_proj", "rowwise"),
        ModuleRule("model.layers.*.mlp.experts.gate_up_proj", "grouped_gemm"),
        ModuleRule("model.layers.*.mlp.experts.down_proj", "grouped_gemm"),
        ModuleRule("lm_head", "colwise"),
    )
    P = ParameterParallelism
    placements = (
        qwen_attention_placements()
        + (
            ParameterPlacement("model.layers.*.mlp.gate.weight", P.REPLICATED),
            ParameterPlacement(
                "model.layers.*.mlp.experts.gate_up_proj", P.EXPERT, 0
            ),
            ParameterPlacement("model.layers.*.mlp.experts.down_proj", P.EXPERT, 0),
        )
        + qwen_replicated_placements()
    )
    return ModelParallelPlan(
        model_type=config.model_type,
        tensor_parallel_size=tp_size,
        expert_parallel_size=ep_size,
        expert_tensor_parallel_size=expert_tp_size,
        module_rules=rules,
        parameter_placements=placements,
        capabilities=ModelParallelCapabilities(
            expert_parallel=True,
            fsdp_composition=True,
        ),
    )

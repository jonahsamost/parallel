from __future__ import annotations

from ..plan import (
    GradientReduction,
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
    sp_size = getattr(pconfig, "sp_size", 1)
    if tp_size <= 1 or ep_size <= 1:
        raise ValueError(
            "Qwen3-MoE folded model parallelism requires tp > 1 and ep > 1"
        )
    if tp_size != ep_size:
        raise ValueError("Qwen3-MoE folded model parallelism requires tp == ep")
    if expert_tp_size != 1:
        raise NotImplementedError("Qwen3-MoE expert tensor parallelism must be 1")
    if sp_size not in (1, tp_size):
        raise ValueError("Qwen3-MoE sequence parallelism requires sp == 1 or sp == tp")
    sequence_parallel = sp_size > 1

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
        ModuleRule("model.layers.*.mlp.experts.gate_up_proj", "grouped_gemm"),
        ModuleRule("model.layers.*.mlp.experts.down_proj", "grouped_gemm"),
        ModuleRule("lm_head", "colwise_sequence" if sequence_parallel else "colwise"),
    )
    P = ParameterParallelism
    placements = (
        qwen_attention_placements()
        + (
            ParameterPlacement(
                "model.layers.*.mlp.gate.weight",
                P.REPLICATED,
                gradient_reduction=(
                    GradientReduction.SUM_TP
                    if sequence_parallel
                    else GradientReduction.NONE
                ),
            ),
            ParameterPlacement(
                "model.layers.*.mlp.experts.gate_up_proj", P.EXPERT, 0
            ),
            ParameterPlacement("model.layers.*.mlp.experts.down_proj", P.EXPERT, 0),
        )
        + qwen_replicated_placements(sequence_parallel)
    )
    return ModelParallelPlan(
        model_type=config.model_type,
        tensor_parallel_size=tp_size,
        expert_parallel_size=ep_size,
        expert_tensor_parallel_size=expert_tp_size,
        sequence_parallel_size=sp_size,
        module_rules=rules,
        parameter_placements=placements,
        capabilities=ModelParallelCapabilities(
            expert_parallel=True,
            sequence_parallel=sequence_parallel,
            fsdp_composition=True,
        ),
    )

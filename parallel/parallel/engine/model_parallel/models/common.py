from __future__ import annotations

from ..plan import GradientReduction, ParameterParallelism, ParameterPlacement


def require_divisible(model_name: str, name: str, value: int, divisor: int) -> None:
    if value % divisor:
        raise ValueError(
            f"{model_name} {name} ({value}) must be divisible by "
            f"tensor-parallel size ({divisor})"
        )


def qwen_attention_placements() -> tuple[ParameterPlacement, ...]:
    P = ParameterParallelism
    return (
        ParameterPlacement("model.embed_tokens.weight", P.VOCAB, 0),
        ParameterPlacement("model.layers.*.self_attn.q_proj.weight", P.COLUMN, 0),
        ParameterPlacement("model.layers.*.self_attn.q_proj.bias", P.COLUMN, 0, False),
        ParameterPlacement("model.layers.*.self_attn.k_proj.weight", P.COLUMN, 0),
        ParameterPlacement("model.layers.*.self_attn.k_proj.bias", P.COLUMN, 0, False),
        ParameterPlacement("model.layers.*.self_attn.v_proj.weight", P.COLUMN, 0),
        ParameterPlacement("model.layers.*.self_attn.v_proj.bias", P.COLUMN, 0, False),
        ParameterPlacement(
            "model.layers.*.self_attn.q_norm.weight",
            P.REPLICATED,
            gradient_reduction=GradientReduction.SUM_TP,
        ),
        ParameterPlacement(
            "model.layers.*.self_attn.k_norm.weight",
            P.REPLICATED,
            gradient_reduction=GradientReduction.SUM_TP,
        ),
        ParameterPlacement("model.layers.*.self_attn.o_proj.weight", P.ROW, 1),
        ParameterPlacement(
            "model.layers.*.self_attn.o_proj.bias", P.REPLICATED, None, False
        ),
        ParameterPlacement("lm_head.weight", P.VOCAB, 0, False),
    )


def qwen_replicated_placements() -> tuple[ParameterPlacement, ...]:
    P = ParameterParallelism
    return (
        ParameterPlacement("model.layers.*.input_layernorm.weight", P.REPLICATED),
        ParameterPlacement(
            "model.layers.*.post_attention_layernorm.weight", P.REPLICATED
        ),
        ParameterPlacement("model.norm.weight", P.REPLICATED),
        # Rotary embedding buffers do not appear in named_parameters.
    )

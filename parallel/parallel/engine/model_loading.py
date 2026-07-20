from __future__ import annotations

from transformers import AutoConfig, AutoModelForCausalLM

from .model_parallel import build_model_parallel_plan
from .model_parallel.tensor_parallel import load_tensor_parallel_causal_lm


def load_causal_lm(cfg, pconfig):
    """Load a model, streaming model-parallel shards from meta initialization."""
    config = AutoConfig.from_pretrained(cfg.model.name)
    if getattr(config, "model_type", None) == "qwen3_moe":
        config.output_router_logits = getattr(
            cfg.model, "output_router_logits", True
        )

    if not pconfig.model_parallel_enabled:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            config=config,
            device_map=None,
            attn_implementation="sdpa",
        )
        return model, None

    plan = build_model_parallel_plan(config, pconfig)
    plan.validate_runtime(pconfig)
    model = load_tensor_parallel_causal_lm(
        cfg.model.name,
        config=config,
        pconfig=pconfig,
        plan=plan,
        dtype="auto",
        attn_implementation="sdpa",
    )
    return model, plan

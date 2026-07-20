from __future__ import annotations

from collections.abc import Callable

from .plan import ModelParallelPlan


PlanBuilder = Callable[[object, object], ModelParallelPlan]
_BUILDERS: dict[str, PlanBuilder] = {}


def register_model_parallel_plan(model_type: str):
    def decorator(builder: PlanBuilder) -> PlanBuilder:
        if model_type in _BUILDERS:
            raise RuntimeError(f"Duplicate model-parallel plan for {model_type}")
        _BUILDERS[model_type] = builder
        return builder

    return decorator


def build_model_parallel_plan(config, pconfig) -> ModelParallelPlan:
    model_type = getattr(config, "model_type", None)
    builder = _BUILDERS.get(model_type)
    if builder is None:
        supported = ", ".join(sorted(_BUILDERS))
        raise NotImplementedError(
            f"Model parallelism does not support model_type={model_type!r}; "
            f"supported model types: {supported}"
        )
    return builder(config, pconfig)


def supported_model_types() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))

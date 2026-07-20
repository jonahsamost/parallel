from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class ParameterParallelism(str, Enum):
    REPLICATED = "replicated"
    COLUMN = "column"
    ROW = "row"
    VOCAB = "vocab"
    EXPERT = "expert"


class GradientReduction(str, Enum):
    NONE = "none"
    SUM_TP = "sum_tp"


@dataclass(frozen=True)
class ModuleRule:
    pattern: str
    style: str


@dataclass(frozen=True)
class ParameterPlacement:
    pattern: str
    parallelism: ParameterParallelism
    shard_dim: int | None = None
    required: bool = True
    gradient_reduction: GradientReduction = GradientReduction.NONE

    def matches(self, name: str) -> bool:
        return fnmatch.fnmatchcase(name, self.pattern)


@dataclass(frozen=True)
class ModelParallelCapabilities:
    tensor_parallel: bool = True
    expert_parallel: bool = False
    expert_tensor_parallel: bool = False
    sequence_parallel: bool = False
    fsdp_composition: bool = False
    portable_checkpoint: bool = True
    sharded_checkpoint: bool = True


@dataclass(frozen=True)
class ModelParallelPlan:
    model_type: str
    tensor_parallel_size: int
    expert_parallel_size: int
    expert_tensor_parallel_size: int
    module_rules: tuple[ModuleRule, ...]
    parameter_placements: tuple[ParameterPlacement, ...]
    capabilities: ModelParallelCapabilities = field(
        default_factory=ModelParallelCapabilities
    )

    @property
    def size(self) -> int:
        return self.tensor_parallel_size

    @property
    def attention_tp_size(self) -> int:
        return self.tensor_parallel_size

    @property
    def hf_plan(self) -> dict[str, str]:
        return {rule.pattern: rule.style for rule in self.module_rules}

    def placement_for_parameter(self, name: str) -> ParameterPlacement:
        for placement in self.parameter_placements:
            if placement.matches(name):
                return placement
        raise KeyError(f"Model-parallel plan does not classify parameter {name!r}")

    def validate_parameter_coverage(self, names: Iterable[str]) -> None:
        names = tuple(names)
        unclassified = []
        for name in names:
            try:
                self.placement_for_parameter(name)
            except KeyError:
                unclassified.append(name)
        if unclassified:
            raise RuntimeError(
                "Model-parallel plan leaves parameters unclassified: "
                + ", ".join(unclassified)
            )

        missing_patterns = [
            placement.pattern
            for placement in self.parameter_placements
            if placement.required
            and not any(placement.matches(name) for name in names)
        ]
        if missing_patterns:
            raise RuntimeError(
                "Model-parallel plan expected parameters matching: "
                + ", ".join(missing_patterns)
            )

    def checkpoint_metadata(self) -> dict:
        return {
            "model_type": self.model_type,
            "tensor_parallel_size": self.tensor_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "expert_tensor_parallel_size": self.expert_tensor_parallel_size,
            "parameters": [
                {
                    "pattern": placement.pattern,
                    "parallelism": placement.parallelism.value,
                    "shard_dim": placement.shard_dim,
                    "required": placement.required,
                    "gradient_reduction": placement.gradient_reduction.value,
                }
                for placement in self.parameter_placements
            ],
        }

    def validate_runtime(self, pconfig) -> None:
        if pconfig.dp_shard_size > 1 and not self.capabilities.fsdp_composition:
            raise NotImplementedError(
                f"{self.model_type} model parallelism does not yet compose with "
                "the custom FSDP implementation; set dp_shard=1"
            )
        if (
            self.expert_tensor_parallel_size > 1
            and not self.capabilities.expert_tensor_parallel
        ):
            raise NotImplementedError(
                f"{self.model_type} does not support expert tensor parallelism"
            )

from dataclasses import dataclass, field, asdict
from enum import Enum
import copy
import torch


class DistType(str, Enum):
    NONE = "None"
    MULTI_CPU = "MULTI_CPU"
    MULTI_GPU = "MULTI_GPU"
    DEEPSPEED = "DEEPSPEED"
    FSDP = "FSDP"
    FLAM = "FLAM"


class RNGType(str, Enum):
    TORCH = "torch"
    CUDA = "cuda"
    GENERATOR = "generator"

TORCH_DISTRIBUTED_OPERATION_TYPES = [
    x.value for x in [
        DistType.MULTI_CPU, DistType.MULTI_GPU,
        DistType.DEEPSPEED, DistType.FSDP, DistType.FLAM
    ]    
]


@dataclass
class GradientAccumulationPlugin():
    num_steps: int = field(
        default=None,
        metadata={"help": "The number of steps to accumulate gradients for."},
    )
    adjust_scheduler: bool = field(
        default=True,
        metadata={
            "help": "Whether to adjust the scheduler steps to account for the number of steps being accumulated. Should be `True` if the used scheduler was not adjusted for gradient accumulation."
        },
    )
    sync_with_dataloader: bool = field(
        default=True,
        metadata={
            "help": "Whether to synchronize setting the gradients when at the end of the dataloader. Should only be set to `False` if you know what you're doing."
        },
    )
    sync_each_batch: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize setting the gradients at each data batch. Setting to `True` may reduce memory requirements when using gradient accumulation with distributed training, at expense of speed."
        },
    )

    def to_dict(self):
        return copy.deepcopy(asdict(self))

    def to_kwargs(self):
        """Return attributes that differ from the default instance."""
        default_dict = asdict(GradientAccumulationPlugin())
        this_dict = asdict(self)
        return {k: v for k, v in this_dict.items() if default_dict[k] != v}


@dataclass
class TensorInformation:
    shape: torch.Size
    dtype: torch.dtype

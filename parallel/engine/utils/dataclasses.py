from dataclasses import dataclass, field
from enum import Enum


class DistType(str, Enum):
    NONE = "None"
    MULTI_CPU = "MULTI_CPU"
    MULTI_GPU = "MULTI_GPU"
    DEEPSPEED = "DEEPSPEED"
    FSDP = "FSDP"
    FLAM = "FLAM"

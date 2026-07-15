from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from parallel.engine._dp_param_unit import DPParamUnit
from parallel.state import ParallelConfig, Strategies


def get_fsdp_units(model: nn.Module) -> list[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    return [model]


class FSDPWrapper:
    """ Wraps a model for FSDP """

    def __init__(
        self,
        model: nn.Module,
        pconfig: ParallelConfig,
        cpu_offload: bool = False,
        activation_checkpoint: bool = False,
        checkpoint_every_n: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.pconfig = pconfig
        self.cpu_offload = cpu_offload
        self.activation_checkpoint = activation_checkpoint
        self.checkpoint_every_n = max(checkpoint_every_n, 1)
        self.device = device

        self.units: list[DPParamUnit] = []
        self.dp_shard_group: Optional[dist.ProcessGroup] = None

    def shard_model(self):
        """Walk the model, create DPParamUnits, shard params, wire prefetch."""
        if self.pconfig.dp_shard_size <= 1 or self.pconfig.device_mesh is None:
            return

        self.dp_shard_group = self.pconfig.device_mesh.get_group(Strategies.DP_SHARD)
        submodules = get_fsdp_units(self.model)

        for i, submodule in enumerate(submodules):
            use_ckpt = (
                self.activation_checkpoint and (i % self.checkpoint_every_n == 0)
            )
            unit = DPParamUnit(
                module=submodule,
                dp_shard_group=self.dp_shard_group,
                unit_index=i,
                cpu_offload=self.cpu_offload,
                use_activation_checkpoint=use_ckpt,
                device=self.device,
            )
            unit.shard()
            self.units.append(unit)

        self._wire_prefetch()

    def _wire_prefetch(self):
        """Link each unit to the next unit for forward/backward prefetch."""
        for i, unit in enumerate(self.units):
            if i + 1 < len(self.units):
                unit.next_fwd = self.units[i + 1]
            if i - 1 >= 0:
                unit.next_bwd = self.units[i - 1]

    def get_sharded_params(self):
        """Return parameters for optimizer creation (after sharding). """
        if not self.units:
            return self.model.parameters()

        shards = []
        for unit in self.units:
            shard = unit.flat_shard
            if not isinstance(shard, nn.Parameter):
                shard = nn.Parameter(shard, requires_grad=True)
                unit.flat_shard = shard
            shards.append(shard)
        return shards

    @property
    def is_active(self) -> bool:
        return len(self.units) > 0

    def remove_hooks(self):
        for unit in self.units:
            unit.remove_hooks()

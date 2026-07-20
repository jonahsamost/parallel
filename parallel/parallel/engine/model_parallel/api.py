from __future__ import annotations

import torch

from .checkpoint import ComposedModelParallelCheckpoint, ModelParallelCheckpoint
from .expert_parallel import (
    ExpertPartition,
    ReplicatedTokenExpertParallel,
    verify_expert_modules,
)
from .grad_norm import clip_grad_norm_
from .loss_parallel import loss_parallel_context, vocab_parallel_cross_entropy
from .registry import build_model_parallel_plan
from .tensor_parallel import (
    install_tensor_parallel_gradient_hooks,
    tensor_parallel_mesh,
    verify_tensor_parallel_model,
)


class ModelParallelWrapper:
    """Public orchestration boundary for tensor and expert model parallelism."""

    def __init__(self, model, pconfig, device, plan=None):
        self.model = model
        self.pconfig = pconfig
        self.device = device
        self.plan = plan
        self.mesh = None
        self.group = None
        self.experts = None
        self.expert_runtime = None
        self.checkpoint = None
        self._tp_gradient_hook_handles = []

        if not self.is_active:
            return
        self.mesh = tensor_parallel_mesh(pconfig)
        self.group = self.mesh.get_group()
        if self.plan is None:
            self.plan = build_model_parallel_plan(model.config, pconfig)
        self.plan.validate_runtime(pconfig)
        verify_tensor_parallel_model(model, self.plan, self.mesh)
        self._tp_gradient_hook_handles = install_tensor_parallel_gradient_hooks(
            model, self.plan, self.mesh
        )

        if self.plan.capabilities.expert_parallel:
            self.experts = ExpertPartition(
                model.config.num_experts,
                self.plan.expert_parallel_size,
                self.mesh.get_local_rank(),
            )
            verify_expert_modules(model, self.experts)
            self.expert_runtime = ReplicatedTokenExpertParallel(
                self.experts, self.group
            )
            self.expert_runtime.apply(model)
        self.checkpoint = ModelParallelCheckpoint(
            model, pconfig, self.plan, self.mesh, device
        )

    @property
    def is_active(self) -> bool:
        return self.pconfig.model_parallel_enabled

    def backward_context(self):
        return loss_parallel_context(self.is_active)

    def token_loss(self, logits, labels, reduction: str = "mean"):
        if not self.is_active:
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction=reduction,
            )
        return vocab_parallel_cross_entropy(
            logits,
            labels,
            tp_mesh=self.mesh,
            vocab_size=self.model.config.vocab_size,
            reduction=reduction,
        )

    def training_loss(self, outputs, labels):
        loss = self.token_loss(outputs.logits, labels)
        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is not None:
            loss = loss + self.model.router_aux_loss_coef * aux_loss.to(loss.device)
        return loss

    def clip_grad_norm_(self, parameters, max_norm: float) -> torch.Tensor:
        if not self.is_active:
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        parameter_ids = {id(parameter) for parameter in parameters}
        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if id(parameter) in parameter_ids
        ]
        return clip_grad_norm_(
            named_parameters,
            max_norm,
            plan=self.plan,
            tp_mesh=self.mesh,
            pconfig=self.pconfig,
            device=self.device,
        )

    def full_state_dict(self):
        return self.checkpoint.full_state_dict()

    def load_full_state_dict(self, state_dict, strict: bool = True):
        return self.checkpoint.load_full_state_dict(state_dict, strict=strict)

    def sharded_state_dict(self):
        return self.checkpoint.sharded_state_dict()

    def load_sharded_state_dict(self, state_dict, strict: bool = True):
        return self.checkpoint.load_sharded_state_dict(state_dict, strict=strict)

    def checkpoint_layout(self):
        return self.checkpoint.checkpoint_layout()

    def attach_fsdp(self, fsdp_wrapper) -> None:
        if not self.is_active or not fsdp_wrapper.is_active:
            return
        self.checkpoint = ComposedModelParallelCheckpoint(
            self.model,
            self.pconfig,
            self.plan,
            self.mesh,
            fsdp_wrapper,
            self.device,
        )

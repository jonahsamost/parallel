from __future__ import annotations

import torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES

from ...state import Strategies
from .plan import GradientReduction, ModelParallelPlan


def tensor_parallel_mesh(pconfig):
    mesh = pconfig.device_mesh
    if mesh is None:
        raise RuntimeError("Tensor parallelism requires a DeviceMesh")
    if mesh.ndim == 1:
        return mesh
    if Strategies.TP not in mesh.mesh_dim_names:
        raise RuntimeError("The distributed DeviceMesh has no tp dimension")
    return mesh[Strategies.TP]


def validate_parallel_styles(plan: ModelParallelPlan) -> None:
    unavailable = sorted(
        {rule.style for rule in plan.module_rules if rule.style not in ALL_PARALLEL_STYLES}
    )
    if unavailable:
        raise RuntimeError(
            "The installed Transformers version lacks model-parallel styles: "
            + ", ".join(unavailable)
        )


def load_tensor_parallel_causal_lm(
    model_name: str,
    *,
    config,
    pconfig,
    plan: ModelParallelPlan,
    **from_pretrained_kwargs,
):
    """Meta-initialize and stream only this rank's planned parameter shards."""
    validate_parallel_styles(plan)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_mesh=pconfig.device_mesh,
        tp_plan=plan.hf_plan,
        **from_pretrained_kwargs,
    )
    verify_tensor_parallel_model(model, plan, tensor_parallel_mesh(pconfig))
    return model


def verify_tensor_parallel_model(model, plan: ModelParallelPlan, tp_mesh) -> None:
    """Fail closed when the loaded model does not match its declared plan."""
    loaded_size = getattr(model, "_tp_size", None)
    if loaded_size != plan.tensor_parallel_size:
        raise RuntimeError(
            f"Expected loaded TP size {plan.tensor_parallel_size}, got {loaded_size}"
        )

    parameters = dict(model.named_parameters())
    plan.validate_parameter_coverage(parameters)
    if tp_mesh.size() != plan.tensor_parallel_size:
        raise RuntimeError(
            f"TP mesh size {tp_mesh.size()} does not match plan size "
            f"{plan.tensor_parallel_size}"
        )

    # Validate that every rank has the same local shape. The supported plans use
    # exact divisibility, so uneven shards indicate a plan/load mismatch.
    if dist.is_initialized() and tp_mesh.size() > 1:
        group = tp_mesh.get_group()
        local_shapes = {name: tuple(parameter.shape) for name, parameter in parameters.items()}
        gathered = [None] * tp_mesh.size()
        dist.all_gather_object(gathered, local_shapes, group=group)
        reference_names = set(local_shapes)
        for rank, shapes in enumerate(gathered):
            if set(shapes) != reference_names:
                raise RuntimeError(f"TP rank {rank} has a different parameter set")
        for name in parameters:
            placement = plan.placement_for_parameter(name)
            shapes = [rank_shapes[name] for rank_shapes in gathered]
            if any(shape != shapes[0] for shape in shapes[1:]):
                raise RuntimeError(f"Uneven or inconsistent TP shards for {name}: {shapes}")
            if placement.shard_dim is not None and shapes[0][placement.shard_dim] == 0:
                raise RuntimeError(f"Empty TP shard for {name}")


def install_tensor_parallel_gradient_hooks(model, plan: ModelParallelPlan, tp_mesh):
    """Reduce partial replicated-parameter gradients before FSDP sees them."""
    handles = []
    group = tp_mesh.get_group()

    def sum_tp_gradient(gradient):
        reduced = gradient.contiguous().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
        return reduced

    for name, parameter in model.named_parameters():
        placement = plan.placement_for_parameter(name)
        if placement.gradient_reduction == GradientReduction.SUM_TP:
            handles.append(parameter.register_hook(sum_tp_gradient))
    return handles

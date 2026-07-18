import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from ..state import Strategies
from ..utils import DTYPE_DICT
from .checkpoint import CheckpointManager
from .dp_sharded import FSDPWrapper
from ._dp_sharded_utils import local_tensor



class ParallelEngine:
    def __init__(
        self, model, tokenizer, pconfig, cfg, device,
    ):
        self.tokenizer = tokenizer
        self.pconfig = pconfig
        self.cfg = cfg
        self.device = device

        self.use_amp = cfg.config.use_amp
        if self.use_amp and cfg.config.amp_dtype not in ("bf16", "fp16"):
            raise ValueError("amp_dtype must be either 'bf16' or 'fp16' when AMP is enabled")
        self.amp_dtype = DTYPE_DICT.get(cfg.config.amp_dtype, torch.float32)
        self.grad_scaler = torch.amp.GradScaler(
            pconfig.device_type,
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

        self.fsdp_wrapper = FSDPWrapper(
            model,
            pconfig,
            cpu_offload=cfg.engine.cpu_offload,
            activation_checkpoint=cfg.engine.activation_checkpoint,
            checkpoint_every_n=cfg.engine.checkpoint_every_n,
            device=device,
        )
        self.fsdp_wrapper.shard_model()
        self.model = model
        self.optimizer_params = self.fsdp_wrapper.get_optimizer_params()
        self._sync_initial_state()

        self.optimizer = torch.optim.AdamW(
            self.optimizer_params,
            lr=cfg.model.learning_rate,
            betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2),
            weight_decay=cfg.optim.weight_decay,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.checkpoint = CheckpointManager(self)

    def _sync_initial_state(self):
        """ Ensure each dp replica starts from identical model state """
        if self.pconfig.device_mesh is None:
            return
        for dim in (Strategies.DP_REPLICATE, Strategies.DP_SHARD):
            if dim not in self.pconfig.device_mesh.mesh_dim_names:
                continue
            group = self.pconfig.device_mesh.get_group(dim)
            source_rank = dist.get_global_rank(group, 0)
            tensors = list(self.model.buffers())
            if not self.fsdp_wrapper.is_active:
                tensors += list(self.model.parameters())
            elif dim == Strategies.DP_REPLICATE:
                tensors += self.fsdp_wrapper.get_sharded_params()
            for tensor in tensors:
                value = local_tensor(tensor).detach()
                if value.device.type == self.device.type:
                    dist.broadcast(value, src=source_rank, group=group)
                    continue
                staged_value = value.to(self.device)
                dist.broadcast(staged_value, src=source_rank, group=group)
                value.copy_(staged_value.to(value.device))

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def forward(self, x, y):
        self.sync_buffers()
        with torch.autocast(device_type=self.pconfig.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
            loss = self.model(input_ids=x, labels=y)
        return loss

    def sync_buffers(self):
        if self.pconfig.device_mesh is None:
            return
        buffers = list(self.model.buffers())
        for dim in (Strategies.DP_REPLICATE, Strategies.DP_SHARD):
            if dim not in self.pconfig.device_mesh.mesh_dim_names:
                continue
            group = self.pconfig.device_mesh.get_group(dim)
            source_rank = dist.get_global_rank(group, 0)
            for buffer in buffers:
                dist.broadcast(buffer, src=source_rank, group=group)
    
    def backward(self, loss):
        self.fsdp_wrapper.prepare_backward(loss)
        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        self.fsdp_wrapper.finalize_backward()
    
    def step(self):
        # The normal sequence is:
        #   - Scale loss.
        #   - Run backward, producing scaled gradients.
        #   - Synchronize scaled gradients.
        #   - Unscale gradients.
        #   - Check for inf/nan.
        #   - Clip unscaled gradients.
        #   - Step optimizer if finite.
        #   - Adjust the scale.
        # Synchronize scaled gradients so every replica observes the same
        # overflow before GradScaler decides whether to step.
        self._average_gradients()

        if self.grad_scaler.is_enabled():
            self.grad_scaler.unscale_(self.optimizer)

        grads_are_finite = self._grads_are_finite()
        if grads_are_finite and self.cfg.model.clip_grad_norm > 0.0:
            self.fsdp_wrapper.clip_grad_norm_(self.cfg.model.clip_grad_norm)

        if not grads_are_finite:
            if not self.grad_scaler.is_enabled():
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite gradients encountered")
            new_scale = self.grad_scaler.get_scale() / 2.0
            self.grad_scaler.update(new_scale=new_scale)
        else:
            if self.grad_scaler.is_enabled():
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
    
    def _average_gradients(self):
        if self.pconfig.dp_replicate_size <= 1 or self.pconfig.device_mesh is None:
            return
        dp_replicate_group = self.pconfig.device_mesh.get_group(Strategies.DP_REPLICATE)
        for param in self.optimizer_params:
            has_grad = torch.tensor(
                int(param.grad is not None),
                dtype=torch.int32,
                device=self.device,
            )
            dist.all_reduce(has_grad, op=dist.ReduceOp.MAX, group=dp_replicate_group)
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            grad = local_tensor(param.grad)
            if grad.device.type == self.device.type:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=dp_replicate_group)
                grad.div_(self.pconfig.dp_replicate_size)
            else:
                staged_grad = grad.to(self.device)
                dist.all_reduce(staged_grad, op=dist.ReduceOp.SUM, group=dp_replicate_group)
                staged_grad.div_(self.pconfig.dp_replicate_size)
                grad.copy_(staged_grad.to(grad.device))
            if not has_grad.item():
                param.grad = None

    def _grads_are_finite(self) -> bool:
        finite = torch.ones((), dtype=torch.int32, device=self.device)
        for param in self.optimizer_params:
            grad = param.grad
            if isinstance(grad, DTensor):
                grad = grad._local_tensor
            if grad is not None and not torch.isfinite(grad).all():
                finite.zero_()
                break

        if self.pconfig.device_mesh is not None:
            for dim in (Strategies.DP_REPLICATE, Strategies.DP_SHARD):
                if dim in self.pconfig.device_mesh.mesh_dim_names:
                    group = self.pconfig.device_mesh.get_group(dim)
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
        return bool(finite.item())

    def state_dict(self):
        return self.checkpoint.full_state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.checkpoint.load_full_state_dict(state_dict, strict=strict)

    def sharded_state_dict(self):
        return self.checkpoint.sharded_state_dict()

    def load_sharded_state_dict(self, state_dict, strict: bool = True):
        return self.checkpoint.load_sharded_state_dict(state_dict, strict=strict)

    def save_checkpoint(
        self,
        path,
        *,
        step: int,
        dataloader_state=None,
        eval_dataloader_state=None,
        metadata=None,
    ):
        return self.checkpoint.save(
            path,
            step=step,
            dataloader_state=dataloader_state,
            eval_dataloader_state=eval_dataloader_state,
            metadata=metadata,
        )

    def load_checkpoint(self, path, *, strict: bool = True):
        return self.checkpoint.load(path, strict=strict)

    def save_full_model(self, path):
        return self.checkpoint.save_full_model(path)

    def load_full_model(self, path, *, strict: bool = True):
        return self.checkpoint.load_full_model(path, strict=strict)
        
    def reduce_loss(self, loss):
        loss_log = loss.detach().float().clone()
        if not self.pconfig.dp_enabled or self.pconfig.device_mesh is None:
            return loss_log
        dims = [
            self.pconfig.device_mesh.get_group(d)
            for d in (Strategies.DP_REPLICATE, Strategies.DP_SHARD)
            if d in self.pconfig.device_mesh.mesh_dim_names
        ]
        for dp_group in dims:
            world_size = dist.get_world_size(dp_group)
            dist.all_reduce(loss_log, op=dist.ReduceOp.SUM, group=dp_group)
            loss_log.div_(world_size)
        return loss_log

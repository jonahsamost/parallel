import torch
import torch.distributed as dist

from parallel.state import Strategies
from parallel.utils import DTYPE_DICT


class ParallelEngine:
    def __init__(
        self, model, tokenizer, pconfig, optimizer, cfg
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pconfig = pconfig
        self.optimizer = optimizer
        self.cfg = cfg

        self.use_amp = cfg.config.use_amp
        if self.use_amp:
            assert cfg.config.amp_dtype in ("bf16", "fp16")
        self.amp_dtype = DTYPE_DICT.get(cfg.config.amp_dtype, torch.float32)
        self.grad_scaler = torch.amp.GradScaler(pconfig.device_type, enabled=self.use_amp)
    
    def forward(self, x, y):
        with torch.autocast(device_type=self.pconfig.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
            loss = self.model(input_ids=x, labels=y)
        return loss
    
    def backward(self, loss):
        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self):
        if self.grad_scaler.is_enabled():
            self.grad_scaler.unscale_(self.optimizer)
        
        self._average_gradients()
        
        if self.cfg.model.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.model.clip_grad_norm)

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
        for p in self.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=dp_replicate_group)
        
    def reduce_loss(self, loss):
        loss_log = loss.float()
        if not self.pconfig.dp_enabled or self.pconfig.device_mesh is None:
            return loss_log
        dims = [
            self.pconfig.device_mesh.get_group(d)
            for d in (Strategies.DP_REPLICATE, Strategies.DP_SHARD)
            if d in self.pconfig.device_mesh.mesh_dim_names
        ]
        for dp_group in dims:
            dist.all_reduce(loss_log, op=dist.ReduceOp.AVG, group=dp_group)
        return loss_log


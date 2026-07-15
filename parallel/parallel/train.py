from omegaconf import OmegaConf
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist

from parallel.eval import eval_bpb, get_token_bytes
from parallel.strategies.dp import average_gradients, communicate_rank_loss
from parallel.dataloader import dist_data_loader
from parallel.logging import get_logger
from parallel.tracking import WandBTracker
from parallel.state import ParallelConfig, RuntimeState, Strategies, init_dist
from parallel.utils import DTYPE_DICT, dist_cleanup, is_dist_initialized, load_cfg, model_init_rngs, model_train_rngs

logger = get_logger(__name__)

def main():
    cfg = load_cfg()
    RuntimeState(backend=cfg.config.backend) # init singleton once
    init_dist(cfg)
    pconfig = ParallelConfig(cfg)
    pconfig.set_device_mesh(cfg.config.device_type)
    wandb = WandBTracker(rank=pconfig.rank)
    wandb.start()
    wandb.store_init_config(OmegaConf.to_container(cfg, resolve=True))

    ### rngs
    model_init_rngs(pconfig, seed=cfg.get("seed", 42))

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        device_map=None,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name,)
    logger.info("Collecting token bytes...")
    token_bytes = get_token_bytes(tokenizer)

    logger.info("Model loaded. Compiling...")
    model = torch.compile(model)
    logger.info("Model compiled")

    model_train_rngs(pconfig, seed=cfg.get("seed", 42))

    train_dataloader = dist_data_loader(
        tokenizer, cfg.model.per_device_batch_size, cfg.model.max_seq_length, split="train"
    )
    eval_dataloader = dist_data_loader(
        tokenizer, cfg.model.per_device_batch_size, cfg.model.max_seq_length, split="val"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.model.learning_rate,
        betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2),
        weight_decay=cfg.optim.weight_decay,
    )

    ### mixed precision
    use_amp = cfg.config.use_amp
    if use_amp:
        assert cfg.config.amp_dtype in ("bf16", "fp16")
    amp_dtype = DTYPE_DICT.get(cfg.config.amp_dtype, torch.float32)
    grad_scaler = torch.amp.GradScaler(pconfig.device_type, enabled=use_amp)

    ### train

    device = torch.device(pconfig.device_type)
    model.to(device)
    model.train()
    tokens_per_step = cfg.model.per_device_batch_size * pconfig.dp_replicate_size * cfg.model.max_seq_length
    grad_accum_steps = cfg.config.grad_accum_steps
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, cfg.config.max_steps + 1):
        step_start = time.time()
        total_loss = torch.tensor(0.0, device=device)
        for _ in range(grad_accum_steps):
            x, y, _ = next(train_dataloader)
            with torch.autocast(device_type=pconfig.device_type, dtype=amp_dtype, enabled=use_amp):
                loss = model(input_ids=x, labels=y).loss / grad_accum_steps

            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.detach() 
        
        if grad_scaler.is_enabled():
            grad_scaler.unscale_(optimizer)

        average_gradients(model, pconfig)

        if cfg.model.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.clip_grad_norm)

        if grad_scaler.is_enabled():
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_log = total_loss.float()
        communicate_rank_loss(loss_log, pconfig)

        now = time.time()
        dt = now - step_start
        toks_per_sec = (tokens_per_step * grad_accum_steps) / dt
        if pconfig.is_main_process:
            logger.info(
                f"step {step:6d} | loss {loss_log.item():.4f} "
                f"| lr {optimizer.param_groups[0]['lr']:.2e} "
                f"| tok/s {toks_per_sec:,.0f}"
            )

        wandb.log({
            "train/loss": loss_log.item(),
            "opt/lr": optimizer.param_groups[0]["lr"],
            "speed/eff_tokens_per_sec": toks_per_sec,
            "step": step,
        }, step=step)

        if step % cfg.config.eval_interval == 0 or step == cfg.config.max_steps:
            logger.info("Evaling...")
            bpb = eval_bpb(
                model, eval_dataloader, cfg.config.eval_steps,
                device, pconfig, use_amp, amp_dtype, token_bytes
            )
            wandb.log({"eval/loss": bpb, "step": step}, step=step)
    
    wandb.finish()
    dist_cleanup()



if __name__ == '__main__':
    main()
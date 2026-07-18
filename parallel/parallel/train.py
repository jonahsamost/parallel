from omegaconf import OmegaConf
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dataloader import dist_data_loader
from .engine.engine import ParallelEngine
from .eval import eval_bpb, get_token_bytes
from .logging import get_logger
from .state import ParallelConfig, RuntimeState, init_dist
from .tracking import WandBTracker
from .utils import dist_cleanup, load_cfg

logger = get_logger(__name__)

def main():
    cfg = load_cfg()
    RuntimeState(backend=cfg.config.backend) # init singleton once
    init_dist(cfg)
    pconfig = ParallelConfig(cfg)
    pconfig.set_device_mesh(cfg.config.device_type)
    device = pconfig.device
    wandb = WandBTracker(rank=pconfig.rank)
    wandb.start()
    wandb.store_init_config(OmegaConf.to_container(cfg, resolve=True))

    ### rngs
    pconfig.model_init_rngs(seed=cfg.get("seed", 42))

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        device_map=None,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name,)
    logger.info("Collecting token bytes...")
    token_bytes = get_token_bytes(tokenizer)

    pconfig.model_train_rngs(seed=cfg.get("seed", 42))

    if cfg.engine.compile and pconfig.dp_shard_size > 1:
        raise NotImplementedError(
            "torch.compile is not yet supported with the custom FSDP implementation"
        )
    if pconfig.dp_shard_size <= 1:
        model.to(device)
    model.train()

    pengine = ParallelEngine(
        model, tokenizer, pconfig, cfg, device
    )
    if cfg.engine.compile:
        logger.info("Model loaded. Compiling...")
        model = torch.compile(model)
        # Checkpointing keeps the original module while training uses its
        # compiled wrapper; both reference the same parameters and buffers.
        pengine.model = model
        logger.info("Model compiled")
    resume = None
    if cfg.config.resume_from:
        logger.info(f"Loading checkpoint from {cfg.config.resume_from}")
        resume = pengine.load_checkpoint(cfg.config.resume_from)

    train_dataloader = dist_data_loader(
        tokenizer,
        cfg.model.per_device_batch_size,
        cfg.model.max_seq_length,
        split="train",
        device=device,
        pconfig=pconfig,
        resume_state_dict=(resume or {}).get("dataloader_state"),
    )
    eval_dataloader = dist_data_loader(
        tokenizer,
        cfg.model.per_device_batch_size,
        cfg.model.max_seq_length,
        split="val",
        device=device,
        pconfig=pconfig,
        resume_state_dict=(resume or {}).get("eval_dataloader_state"),
    )

    ### train

    tokens_per_step = (
        cfg.model.per_device_batch_size
        * pconfig.data_world_size
        * cfg.model.max_seq_length
    )
    grad_accum_steps = cfg.config.grad_accum_steps

    start_step = (resume or {}).get("step", 0) + 1
    eval_dataloader_state = (resume or {}).get("eval_dataloader_state")
    for step in range(start_step, cfg.config.max_steps + 1):
        step_start = time.time()
        total_loss = torch.tensor(0.0, device=device)
        dataloader_state = None
        for _ in range(grad_accum_steps):
            x, y, dataloader_state = next(train_dataloader)
            loss = pengine.forward(x, y).loss / grad_accum_steps
            pengine.backward(loss)
            total_loss += loss.detach() 
        
        pengine.step()
        loss_log = pengine.reduce_loss(total_loss)

        now = time.time()
        dt = now - step_start
        toks_per_sec = (tokens_per_step * grad_accum_steps) / dt
        if pconfig.is_main_process:
            logger.info(
                f"step {step:6d} | loss {loss_log.item():.4f} "
                f"| lr {pengine.lr:.2e} "
                f"| tok/s {toks_per_sec:,.0f}"
            )

        wandb.log({
            "train/loss": loss_log.item(),
            "opt/lr": pengine.lr,
            "speed/eff_tokens_per_sec": toks_per_sec,
            "step": step,
        }, step=step)

        if step % cfg.config.eval_interval == 0 or step == cfg.config.max_steps:
            logger.info("Evaling...")
            pengine.sync_buffers()
            bpb, eval_dataloader_state = eval_bpb(
                model, eval_dataloader, cfg.config.eval_steps,
                device, pconfig, pengine.use_amp, pengine.amp_dtype, token_bytes
            )
            wandb.log({"eval/loss": bpb, "step": step}, step=step)

        checkpoint_interval = cfg.config.checkpoint_interval
        if checkpoint_interval > 0 and step % checkpoint_interval == 0:
            checkpoint_path = (
                Path(cfg.config.checkpoint_dir) / f"step-{step:08d}"
            )
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            pengine.save_checkpoint(
                checkpoint_path,
                step=step,
                dataloader_state=dataloader_state,
                eval_dataloader_state=eval_dataloader_state,
            )
    
    wandb.finish()
    dist_cleanup()



if __name__ == '__main__':
    main()

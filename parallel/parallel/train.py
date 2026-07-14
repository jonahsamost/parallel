from parallel.parallel.dataloader import dist_data_loader
from parallel.parallel.logging import get_logger
import torch

from parallel.parallel.state import ParallelConfig, RuntimeState, init_dist
from parallel.parallel.utils import load_cfg, model_init_rngs, model_train_rngs
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = get_logger(__name__)

def main():
    cfg = load_cfg()
    RuntimeState(backend=cfg.config.backend) # init singleton once
    init_dist(cfg)
    pconfig = ParallelConfig(cfg)
    pconfig.set_device_mesh(cfg.config.device_type)

    model_init_rngs(pconfig, seed=cfg.get("seed", 42))

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        device_map="auto",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name,)

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



if __name__ == '__main__':
    main()
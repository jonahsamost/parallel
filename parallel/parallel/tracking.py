import os
from typing import Optional

from parallel.parallel.logging import get_logger


logger = get_logger(__name__)

class WandBTracker:
    def __init__(self, rank: int, project_name: str = None, **kwargs):
        self.rank = rank
        self.project_name = project_name
        self.init_kwargs = kwargs
        assert os.environ.get("WANDB_API_KEY")
    
    @property
    def is_main_rank(self):
        return self.rank == 0
    
    def start(self):
        if self.is_main_rank:
            import wandb
            self.run = wandb.init(project=self.project_name, **self.init_kwargs)
            logger.debug("initialized wandb")
    
    def store_init_config(self, values: dict):
        if self.is_main_rank:
            import wandb
            wandb.config.update(values, allow_val_change=True)
            logger.debug("Stored config to wandb")
    
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        if self.is_main_rank:
            self.run.log(values, step=step, **kwargs)
            logger.debug("logged to wandb")
    
    def finish(self):
        if self.is_main_rank:
            self.run.finish()
            logger.debug("wandb closed")

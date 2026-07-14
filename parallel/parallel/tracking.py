from typing import Optional

from parallel.parallel.logging import get_logger


logger = get_logger(__name__)

class WandBTracker:
    def __init__(self, project_name: str, **kwargs):
        self.project_name = project_name
        self.init_kwargs = kwargs
    
    def start(self):
        import wandb
        self.run = wandb.init(project=self.project_name, **self.init_kwargs)
        logger.debug("initialized wandb")
    
    def store_init_config(self, values: dict):
        import wandb
        wandb.config.update(values, allow_val_change=True)
        logger.debug("Stored config to wandb")
    
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        self.run.log(values, step=step, **kwargs)
        logger.debug("logged to wandb")
    
    def finish(self):
        self.run.finish()
        logger.debug("wandb closed")

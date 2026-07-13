import os
from functools import wraps
from typing import Callable, Optional

from parallel.engine.state import RuntimeState

from .utils.imports import is_wandb_available
from .logging import get_logger


logger = get_logger(__name__)


def on_main_process(function: Callable):
    @wraps(function)
    def execute_on_main_process(self, *args, **kwargs):
        if getattr(self, "main_process_only", False):
            return RuntimeState().on_main_process(function)(self, *args, **kwargs)
        else:
            return function(self, *args, **kwargs)
    return execute_on_main_process

class BaseTracker:
    def __init__(
        self,
        main_process_only: bool = True,
    ):
        self.main_process_only = main_process_only


class WandBTracker(BaseTracker):
    def __init__(
        self,
        project_name: str,
        main_process_only: bool = True,
        **kwargs,
    ):
        super().__init__(main_process_only=main_process_only)
        assert is_wandb_available()
        self.project_name = project_name
        self.init_kwargs = kwargs
    
    def start(self):
        import wandb
        self.run = wandb.init(
            project=self.project_name, **self.init_kwargs
        )
        logger.debug(f"Started WanB project: {self.project_name}")
    
    @on_main_process
    def store_init_config(self, values: dict):
        import wandb
        wandb.config.update(values, allow_val_change=True)
        logger.debug("Stored initial config to WandB")
    
    @on_main_process
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        self.run.log(values, step=step, **kwargs)
        logger.debug("Logged to WandB")
    
    @on_main_process
    def finish(self):
        self.run.finish()
        logger.debug("WandB run finished")
        


import os
from .utils.imports import is_wandb_available
from .logging import get_logger


logger = get_logger(__name__)

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
    
    def log(self):
        ...


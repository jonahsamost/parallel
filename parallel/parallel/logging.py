from parallel.collectives import wait_for_everyone
import torch
import os
import logging

from parallel.state import RuntimeState

class MultiProcAdapter(logging.LoggerAdapter):
    
    def process(self, msg, kwargs):
        msg, kwargs = super().process(msg, kwargs)
        kwargs.setdefault("stacklevel", 2)
        state = RuntimeState()
        msg = f"[RANK {state.rank}] {msg}"
        return msg, kwargs
    
    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            state = RuntimeState()
            msg, kwargs = self.process(msg, kwargs)
            log_all_ranks = kwargs.pop("LOG_ALL_RANKS", False)
            if log_all_ranks and state.world_size > 1:
                for i in range(state.world_size):
                    if i == state.rank:
                        self.logger.log(level, msg, *args, **kwargs)
                    wait_for_everyone(state)
            elif state.can_log:
                self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: str, log_level: str | None = None):
    if log_level is None:
        log_level = os.environ.get("PARALLEL_LOG_LEVEL", None)
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())
    return MultiProcAdapter(logger, {})

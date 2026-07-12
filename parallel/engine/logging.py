from __future__ import annotations

import logging
import os
from enum import Enum

from parallel.engine.state import RuntimeState

from .utils.environment import get_log_level


class MultiProcLogger(logging.LoggerAdapter):
    @staticmethod
    def _should_log(main_process_only):
        state = RuntimeState()
        return not main_process_only or (main_process_only and state.is_main_process)
    
    def process(self, msg, kwargs):
        msg, kwargs = super().process(msg, kwargs)
        kwargs.setdefault("stacklevel", 2)
        state = RuntimeState()
        msg = f"[RANK {state.process_idx}] {msg}"
        return msg, kwargs
    
    def log(self, level, msg, *args, **kwargs):
        ...


def get_logger(
    name: str,
    log_level: str | None = None,
):
    if log_level is None:
        log_level = get_log_level()
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())
    return MultiProcLogger(logger, {})
    
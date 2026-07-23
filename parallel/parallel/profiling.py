from __future__ import annotations

import logging
from contextlib import ContextDecorator
from time import perf_counter
from typing import Any

from .logging import get_logger


_DEFAULT_LOGGER = get_logger(__name__)


class _Profile(ContextDecorator):
    def __init__(
        self,
        name: str,
        *,
        logger: logging.Logger | logging.LoggerAdapter,
        level: int,
        enabled: bool,
        synchronize: bool,
        device: Any,
        use_torch_profiler: bool,
        use_cuda_events: bool,
    ) -> None:
        if not name:
            raise ValueError("profile name cannot be empty")
        self.name = name
        self.logger = logger
        self.level = level
        self.enabled = enabled
        self.synchronize = synchronize
        self.device = device
        self.use_torch_profiler = use_torch_profiler
        self.use_cuda_events = use_cuda_events
        self._start: float | None = None
        self._cuda_start: Any = None
        self._cuda_end: Any = None
        self._torch_profiler: Any = None
        self._record_function: Any = None
        self._torch_profiler_sort_key = "self_cpu_time_total"

    def _recreate_cm(self) -> _Profile:
        # Decorated functions can be recursive or run concurrently, so every
        # invocation needs its own start time.
        return type(self)(
            self.name,
            logger=self.logger,
            level=self.level,
            enabled=self.enabled,
            synchronize=self.synchronize,
            device=self.device,
            use_torch_profiler=self.use_torch_profiler,
            use_cuda_events=self.use_cuda_events,
        )

    def _cuda_device(self, torch: Any) -> Any:
        device = torch.device(self.device) if self.device is not None else None
        if device is not None and device.type != "cuda":
            return None
        return device

    def _synchronize_cuda(self) -> None:
        if not self.synchronize:
            return

        import torch

        device = self._cuda_device(torch)
        if self.device is not None and device is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _start_torch_profiler(self, torch: Any) -> None:
        if not self.use_torch_profiler:
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        device = self._cuda_device(torch)
        if torch.cuda.is_available() and (self.device is None or device is not None):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            self._torch_profiler_sort_key = "self_cuda_time_total"

        self._torch_profiler = torch.profiler.profile(activities=activities)
        self._torch_profiler.__enter__()
        self._record_function = torch.profiler.record_function(self.name)
        self._record_function.__enter__()

    def _start_cuda_events(self, torch: Any) -> None:
        if not self.use_cuda_events:
            return

        device = self._cuda_device(torch)
        stream = torch.cuda.current_stream(device)
        self._cuda_start = torch.cuda.Event(enable_timing=True)
        self._cuda_end = torch.cuda.Event(enable_timing=True)
        self._cuda_start.record(stream)

    def _validate_cuda_events(self, torch: Any) -> None:
        if not self.use_cuda_events:
            return
        if self.device is not None and self._cuda_device(torch) is None:
            raise ValueError("use_cuda_events requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("use_cuda_events requires CUDA to be available")

    def __enter__(self) -> _Profile:
        if not self.enabled:
            return self
        self._synchronize_cuda()
        if self.use_torch_profiler or self.use_cuda_events:
            import torch

            self._validate_cuda_events(torch)
            self._start_torch_profiler(torch)
            self._start_cuda_events(torch)
        self._start = perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        if not self.enabled:
            return False
        if self._cuda_end is not None:
            import torch

            self._cuda_end.record(torch.cuda.current_stream(self._cuda_device(torch)))
        self._synchronize_cuda()
        if self._start is None:
            raise RuntimeError("profile context exited before it was entered")
        elapsed_ms = (perf_counter() - self._start) * 1_000

        torch_profiler_table = None
        if self._record_function is not None:
            self._record_function.__exit__(*exc_info)
        if self._torch_profiler is not None:
            self._torch_profiler.__exit__(*exc_info)
            torch_profiler_table = self._torch_profiler.key_averages().table(
                sort_by=self._torch_profiler_sort_key,
                row_limit=10,
            )

        cuda_elapsed_ms = None
        if self._cuda_start is not None and self._cuda_end is not None:
            self._cuda_end.synchronize()
            cuda_elapsed_ms = self._cuda_start.elapsed_time(self._cuda_end)

        self.logger.log(self.level, "[PROFILE] %s: %.3f ms", self.name, elapsed_ms)
        if torch_profiler_table is not None:
            self.logger.log(
                self.level,
                "[PROFILE] %s torch profiler:\n%s",
                self.name,
                torch_profiler_table,
            )
        if cuda_elapsed_ms is not None:
            self.logger.log(
                self.level,
                "[PROFILE] %s CUDA event: %.3f ms",
                self.name,
                cuda_elapsed_ms,
            )
        return False


def profile(
    name: str,
    *,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
    level: int = logging.INFO,
    enabled: bool = True,
    synchronize: bool = False,
    device: Any = None,
    use_torch_profiler: bool = False,
    use_cuda_events: bool = False,
) -> _Profile:
    """Time a block or decorated function and log its duration.

    Set ``enabled=False`` to make the context manager or decorator a no-op
    without changing the profiled code.

    Set ``synchronize=True`` when measuring asynchronous CUDA work. This waits
    for the device before and after the operation, giving an accurate duration
    at the cost of removing overlap with other CUDA work.

    Set ``use_cuda_events=True`` to additionally measure work enqueued on the
    current CUDA stream; the end event is synchronized when the block exits.
    Set ``use_torch_profiler=True`` to log a ten-row operator summary for the
    block. Both modes have more overhead than the default wall-clock timer and
    are intended for focused investigations.
    """
    return _Profile(
        name,
        logger=logger if logger is not None else _DEFAULT_LOGGER,
        level=level,
        enabled=enabled,
        synchronize=synchronize,
        device=device,
        use_torch_profiler=use_torch_profiler,
        use_cuda_events=use_cuda_events,
    )

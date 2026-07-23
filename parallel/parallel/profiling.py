from __future__ import annotations

import logging
import math
from collections import defaultdict
from contextlib import ContextDecorator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .logging import get_logger


_DEFAULT_LOGGER = get_logger(__name__)
_COLLECTIVE_PHASE = "other"


def _tensor_bytes(tensor: Any) -> int:
    return tensor.numel() * tensor.element_size()


@dataclass
class _CollectiveRecord:
    operation: str
    phase: str
    mode: str
    detail: str
    payload_bytes: int
    cpu_ms: float
    cuda_start: Any = None
    cuda_end: Any = None
    cuda_ms: float | None = None


class _CollectiveProfile:
    def __init__(
        self,
        operation: str,
        *,
        mode: str,
        detail: str,
        payload_bytes: int,
        device: Any,
        defer_cuda_end: bool,
    ) -> None:
        self.operation = operation
        self.mode = mode
        self.detail = detail
        self.payload_bytes = payload_bytes
        self.device = device
        self.defer_cuda_end = defer_cuda_end
        self.phase = "other"
        self._start = None
        self._cuda_start = None
        self._cuda_end = None
        self._record_function = None

    def __enter__(self):
        if not _COLLECTIVE_PROFILER.recording:
            return self
        import torch

        self.phase = _COLLECTIVE_PHASE
        label = ".".join(
            part
            for part in (
                "collective",
                self.phase,
                self.operation,
                self.mode,
                self.detail,
            )
            if part
        )
        self._record_function = torch.profiler.record_function(label)
        self._record_function.__enter__()
        device = torch.device(self.device) if self.device is not None else None
        if (
            device is not None
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            stream = torch.cuda.current_stream(device)
            self._cuda_start = torch.cuda.Event(enable_timing=True)
            self._cuda_end = torch.cuda.Event(enable_timing=True)
            self._cuda_start.record(stream)
        self._start = perf_counter()
        return self

    def __exit__(self, *exc_info):
        if self._start is None:
            return False
        if self._cuda_end is not None and not self.defer_cuda_end:
            import torch

            self._cuda_end.record(torch.cuda.current_stream(self.device))
        cpu_ms = (perf_counter() - self._start) * 1_000
        if self._record_function is not None:
            self._record_function.__exit__(*exc_info)
        self._record = _CollectiveRecord(
            operation=self.operation,
            phase=self.phase,
            mode=self.mode,
            detail=self.detail,
            payload_bytes=self.payload_bytes,
            cpu_ms=cpu_ms,
            cuda_start=self._cuda_start,
            cuda_end=self._cuda_end,
        )
        if not self.defer_cuda_end:
            _COLLECTIVE_PROFILER.add(self._record)
        return False

    def complete(self) -> None:
        """Finish a CUDA interval whose collective was launched asynchronously."""
        if not self.defer_cuda_end:
            raise RuntimeError("Only deferred collective profiles can be completed")
        if self._start is None:
            return
        if not hasattr(self, "_record"):
            raise RuntimeError("Deferred collective profile has not been entered")
        if self._cuda_end is not None:
            import torch

            self._cuda_end.record(torch.cuda.current_stream(self.device))
        _COLLECTIVE_PROFILER.add(self._record)
        self.defer_cuda_end = False


class CollectiveProfiler:
    """Deferred CUDA-event timing and per-step collective aggregation."""

    def __init__(self) -> None:
        self.enabled = False
        self.recording = False
        self.device = None
        self.records: list[_CollectiveRecord] = []

    def configure(self, *, enabled: bool, device: Any = None) -> None:
        self.enabled = enabled
        self.device = device
        self.recording = False
        self.records.clear()

    def start_step(self) -> None:
        self.records.clear()
        self.recording = self.enabled

    def add(self, record: _CollectiveRecord) -> None:
        if self.recording:
            self.records.append(record)

    def _resolve_cuda_events(self) -> None:
        cuda_records = [record for record in self.records if record.cuda_end is not None]
        if not cuda_records:
            return
        import torch

        torch.cuda.synchronize(self.device)
        for record in cuda_records:
            record.cuda_ms = record.cuda_start.elapsed_time(record.cuda_end)

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = math.ceil((len(ordered) - 1) * percentile)
        return ordered[index]

    @staticmethod
    def _format_bytes(value: int) -> str:
        if value >= 1024 ** 3:
            return f"{value / 1024 ** 3:.2f}GiB"
        return f"{value / 1024 ** 2:.1f}MiB"

    def _local_payload(self, rank: int) -> list[dict[str, Any]]:
        return [
            {
                "rank": rank,
                "operation": record.operation,
                "phase": record.phase,
                "mode": record.mode,
                "detail": record.detail,
                "payload_bytes": record.payload_bytes,
                "cpu_ms": record.cpu_ms,
                "cuda_ms": record.cuda_ms,
            }
            for record in self.records
        ]

    def finish_step(self) -> list[str]:
        if not self.enabled:
            return []
        self.recording = False
        self._resolve_cuda_events()

        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        local = self._local_payload(rank)
        if dist.is_initialized():
            gathered: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            records = [record for rank_records in gathered for record in rank_records]
        else:
            records = local

        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[
                (record["phase"], record["operation"], record["mode"])
            ].append(record)

        lines = []
        for (phase, operation, mode), group_records in sorted(grouped.items()):
            by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for record in group_records:
                by_rank[record["rank"]].append(record)
            calls = [len(rank_records) for rank_records in by_rank.values()]
            payloads = [
                sum(record["payload_bytes"] for record in rank_records)
                for rank_records in by_rank.values()
            ]
            cuda_values = [
                record["cuda_ms"]
                for record in group_records
                if record["cuda_ms"] is not None
            ]
            timing_key = "cuda_ms" if cuda_values else "cpu_ms"
            timings = [record[timing_key] for record in group_records]
            totals = [
                sum(record[timing_key] for record in rank_records)
                for rank_records in by_rank.values()
            ]
            slowest = max(group_records, key=lambda record: record[timing_key])
            call_text = (
                str(calls[0])
                if min(calls) == max(calls)
                else f"{min(calls)}-{max(calls)}"
            )
            lines.append(
                f"[COLLECTIVE] phase={phase} op={operation} mode={mode} "
                f"calls/rank={call_text} "
                f"payload/max-rank={self._format_bytes(max(payloads))} "
                f"{timing_key[:-3]} total/max-rank={max(totals):.3f}ms "
                f"p50={self._percentile(timings, 0.50):.3f}ms "
                f"p95={self._percentile(timings, 0.95):.3f}ms "
                f"max={slowest[timing_key]:.3f}ms "
                f"slowest=rank{slowest['rank']}:{slowest['detail']} "
                f"({self._format_bytes(slowest['payload_bytes'])})"
            )
        self.records.clear()
        return lines


_COLLECTIVE_PROFILER = CollectiveProfiler()


def configure_collective_profiling(*, enabled: bool, device: Any = None) -> None:
    _COLLECTIVE_PROFILER.configure(enabled=enabled, device=device)


def start_collective_profile_step() -> None:
    _COLLECTIVE_PROFILER.start_step()


def finish_collective_profile_step() -> list[str]:
    return _COLLECTIVE_PROFILER.finish_step()


def scheduled_torch_profiler(
    *,
    enabled: bool,
    trace_dir: str | Path,
    rank: int,
    warmup_steps: int = 1,
    active_steps: int = 1,
):
    """Create a scheduled CPU/CUDA profiler with one trace file per rank."""
    if not enabled:
        return None
    if warmup_steps < 0:
        raise ValueError("torch profiler warmup steps cannot be negative")
    if active_steps < 1:
        raise ValueError("torch profiler active steps must be positive")

    import torch

    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=warmup_steps,
            active=active_steps,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(trace_dir),
            worker_name=f"rank{rank}",
            use_gzip=True,
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
    )


@contextmanager
def collective_phase(phase: str):
    global _COLLECTIVE_PHASE
    previous = _COLLECTIVE_PHASE
    _COLLECTIVE_PHASE = phase
    try:
        yield
    finally:
        _COLLECTIVE_PHASE = previous


def collective_profile(
    operation: str,
    *,
    value: Any = None,
    payload_bytes: int | None = None,
    mode: str = "",
    detail: str = "",
    device: Any = None,
    defer_cuda_end: bool = False,
) -> _CollectiveProfile:
    if payload_bytes is None:
        payload_bytes = _tensor_bytes(value) if value is not None else 0
    if device is None and value is not None:
        device = value.device
    return _CollectiveProfile(
        operation,
        mode=mode,
        detail=detail,
        payload_bytes=payload_bytes,
        device=device,
        defer_cuda_end=defer_cuda_end,
    )


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

import logging
from unittest.mock import MagicMock, Mock, call

import pytest
import torch

from parallel.parallel import profile
from parallel.parallel import profiling


def test_profile_times_context_manager(monkeypatch):
    logger = Mock(spec=logging.Logger)
    monkeypatch.setattr(profiling, "perf_counter", Mock(side_effect=[1.0, 1.0125]))

    with profile("copy shard", logger=logger):
        pass

    assert logger.log.call_args == call(
        logging.INFO,
        "[PROFILE] %s: %.3f ms",
        "copy shard",
        pytest.approx(12.5),
    )


def test_profile_decorates_function(monkeypatch):
    logger = Mock(spec=logging.Logger)
    monkeypatch.setattr(profiling, "perf_counter", Mock(side_effect=[2.0, 2.005]))

    @profile("add", logger=logger, level=logging.DEBUG)
    def add(left, right):
        return left + right

    assert add(2, 3) == 5
    assert add.__name__ == "add"
    assert logger.log.call_args == call(
        logging.DEBUG,
        "[PROFILE] %s: %.3f ms",
        "add",
        pytest.approx(5.0),
    )


def test_profile_logs_when_block_raises(monkeypatch):
    logger = Mock(spec=logging.Logger)
    monkeypatch.setattr(profiling, "perf_counter", Mock(side_effect=[3.0, 3.001]))

    with pytest.raises(ValueError, match="boom"):
        with profile("failing block", logger=logger):
            raise ValueError("boom")

    logger.log.assert_called_once()


def test_profile_rejects_empty_name():
    with pytest.raises(ValueError, match="cannot be empty"):
        profile("")


def test_profile_can_be_disabled_for_context_and_decorator(monkeypatch):
    logger = Mock(spec=logging.Logger)
    timer = Mock()
    monkeypatch.setattr(profiling, "perf_counter", timer)

    with profile(
        "disabled context",
        logger=logger,
        enabled=False,
        synchronize=True,
        device="cuda:0",
        use_cuda_events=True,
        use_torch_profiler=True,
    ):
        pass

    @profile("disabled decorator", logger=logger, enabled=False)
    def add(left, right):
        return left + right

    assert add(2, 3) == 5
    timer.assert_not_called()
    logger.log.assert_not_called()


def test_profile_uses_cuda_events(monkeypatch):
    logger = Mock(spec=logging.Logger)
    start_event = Mock()
    end_event = Mock()
    start_event.elapsed_time.return_value = 4.25
    event_factory = Mock(side_effect=[start_event, end_event])
    stream = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=True))
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=stream))
    monkeypatch.setattr(torch.cuda, "Event", event_factory)

    with profile(
        "cuda work",
        logger=logger,
        device="cuda:0",
        use_cuda_events=True,
    ):
        pass

    start_event.record.assert_called_once_with(stream)
    end_event.record.assert_called_once_with(stream)
    end_event.synchronize.assert_called_once_with()
    assert logger.log.call_args == call(
        logging.INFO,
        "[PROFILE] %s CUDA event: %.3f ms",
        "cuda work",
        4.25,
    )


def test_profile_uses_torch_profiler(monkeypatch):
    logger = Mock(spec=logging.Logger)
    profiler_context = MagicMock()
    profiler_context.key_averages.return_value.table.return_value = "operator table"
    record_context = MagicMock()
    profiler_factory = Mock(return_value=profiler_context)
    record_factory = Mock(return_value=record_context)
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    monkeypatch.setattr(torch.profiler, "profile", profiler_factory)
    monkeypatch.setattr(torch.profiler, "record_function", record_factory)

    with profile("model", logger=logger, use_torch_profiler=True):
        pass

    profiler_factory.assert_called_once_with(
        activities=[torch.profiler.ProfilerActivity.CPU]
    )
    record_factory.assert_called_once_with("model")
    profiler_context.__enter__.assert_called_once_with()
    record_context.__enter__.assert_called_once_with()
    record_context.__exit__.assert_called_once_with(None, None, None)
    profiler_context.__exit__.assert_called_once_with(None, None, None)
    assert logger.log.call_args == call(
        logging.INFO,
        "[PROFILE] %s torch profiler:\n%s",
        "model",
        "operator table",
    )


def test_collective_profiler_aggregates_without_per_call_logging():
    profiling.configure_collective_profiling(enabled=True, device="cpu")
    profiling.start_collective_profile_step()
    try:
        with profiling.collective_phase("forward"):
            with profiling.collective_profile(
                "fsdp_all_gather",
                payload_bytes=4 * 1024 ** 2,
                mode="prefetch",
                detail="unit3:model.layers.1.self_attn.q_proj.weight",
                device="cpu",
            ):
                pass
            with profiling.collective_profile(
                "fsdp_all_gather",
                payload_bytes=8 * 1024 ** 2,
                mode="prefetch",
                detail="unit4:model.layers.1.self_attn.k_proj.weight",
                device="cpu",
            ):
                pass

        lines = profiling.finish_collective_profile_step()
    finally:
        profiling.configure_collective_profiling(enabled=False)

    assert len(lines) == 1
    assert "phase=forward" in lines[0]
    assert "op=fsdp_all_gather" in lines[0]
    assert "mode=prefetch" in lines[0]
    assert "calls/rank=2" in lines[0]
    assert "payload/max-rank=12.0MiB" in lines[0]


def test_collective_profiler_defers_async_record_until_complete():
    profiling.configure_collective_profiling(enabled=True, device="cpu")
    profiling.start_collective_profile_step()
    try:
        with profiling.collective_phase("backward"):
            measurement = profiling.collective_profile(
                "fsdp_reduce_scatter",
                payload_bytes=16 * 1024 ** 2,
                mode="overlapped",
                detail="unit1:model.layers.0.weight",
                device="cpu",
                defer_cuda_end=True,
            )
            with measurement:
                pass
            assert profiling._COLLECTIVE_PROFILER.records == []
            measurement.complete()

        lines = profiling.finish_collective_profile_step()
    finally:
        profiling.configure_collective_profiling(enabled=False)

    assert len(lines) == 1
    assert "phase=backward" in lines[0]
    assert "op=fsdp_reduce_scatter" in lines[0]


def test_scheduled_torch_profiler_warms_up_then_captures(monkeypatch, tmp_path):
    schedule = Mock()
    handler = Mock()
    profiler_instance = Mock()
    schedule_factory = Mock(return_value=schedule)
    handler_factory = Mock(return_value=handler)
    profiler_factory = Mock(return_value=profiler_instance)
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    monkeypatch.setattr(torch.profiler, "schedule", schedule_factory)
    monkeypatch.setattr(
        torch.profiler, "tensorboard_trace_handler", handler_factory
    )
    monkeypatch.setattr(torch.profiler, "profile", profiler_factory)

    result = profiling.scheduled_torch_profiler(
        enabled=True,
        trace_dir=tmp_path / "traces",
        rank=3,
        warmup_steps=1,
        active_steps=1,
    )

    assert result is profiler_instance
    schedule_factory.assert_called_once_with(
        wait=0, warmup=1, active=1, repeat=1
    )
    handler_factory.assert_called_once_with(
        str(tmp_path / "traces"),
        worker_name="rank3",
        use_gzip=True,
    )
    profiler_factory.assert_called_once_with(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
    )


def test_scheduled_torch_profiler_can_be_disabled(tmp_path):
    result = profiling.scheduled_torch_profiler(
        enabled=False,
        trace_dir=tmp_path / "traces",
        rank=0,
    )
    assert result is None
    assert not (tmp_path / "traces").exists()

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

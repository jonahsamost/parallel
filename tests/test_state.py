import pytest
from omegaconf import OmegaConf

from parallel.parallel.state import ParallelConfig, RuntimeState


def _config(**overrides):
    parallel = {
        "dp_replicate": 1,
        "dp_shard": 1,
        "tp": 1,
        "cp": 1,
        "sp": 1,
        "ep": 1,
        "pp": 1,
    }
    parallel.update(overrides)
    return OmegaConf.create({"parallel": parallel})


def test_runtime_state_defaults_to_single_process(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    RuntimeState._state = {}
    assert RuntimeState().world_size == 1


def test_parallel_config_rejects_mesh_world_size_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        ParallelConfig(_config(dp_shard=2))


def test_parallel_config_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="positive integers"):
        ParallelConfig(_config(dp_replicate=0))

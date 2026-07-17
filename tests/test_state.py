import pytest
import torch
from omegaconf import OmegaConf

from parallel.parallel.state import ParallelConfig, RuntimeState, Strategies


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


class _FakeMesh:
    mesh_dim_names = (Strategies.DP_REPLICATE, Strategies.DP_SHARD)

    def __init__(self, replicate_rank, shard_rank):
        self.ranks = {
            Strategies.DP_REPLICATE: replicate_rank,
            Strategies.DP_SHARD: shard_rank,
        }

    def get_local_rank(self, dim_name):
        return self.ranks[dim_name]


def _hybrid_config(replicate_rank=1, shard_rank=2):
    config = ParallelConfig.__new__(ParallelConfig)
    config.dp_replicate_size = 2
    config.dp_shard_size = 4
    config.device_mesh = _FakeMesh(replicate_rank, shard_rank)
    config.device_type = "cpu"
    config.local_rank = 0
    config._device = None
    return config


def test_data_rank_covers_replicate_and_shard_dimensions():
    config = _hybrid_config(replicate_rank=1, shard_rank=2)

    assert config.data_rank == 6
    assert config.data_world_size == 8


def test_training_rng_uses_full_data_parallel_rank(monkeypatch):
    config = _hybrid_config(replicate_rank=1, shard_rank=2)
    random_seed = []
    numpy_seed = []
    torch_seed = []
    monkeypatch.setattr("parallel.parallel.state.random.seed", random_seed.append)
    monkeypatch.setattr("parallel.parallel.state.np.random.seed", numpy_seed.append)
    monkeypatch.setattr("parallel.parallel.state.torch.manual_seed", torch_seed.append)

    config.model_train_rngs(seed=10)

    assert random_seed == [16]
    assert numpy_seed == [16]
    assert torch_seed == [6010]


def test_device_is_exposed_as_a_property():
    config = _hybrid_config()

    assert config.device == torch.device("cpu")

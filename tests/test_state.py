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


def test_parallel_config_allows_tensor_parallel_mesh_dimension(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr("parallel.parallel.state.is_dist_initialized", lambda: True)
    monkeypatch.setattr("parallel.parallel.state.dist.get_rank", lambda: 0)
    monkeypatch.setattr("parallel.parallel.state.dist.get_world_size", lambda: 8)

    config = ParallelConfig(_config(tp=4, ep=4, dp_shard=2))

    assert config.get_mesh_layout() == (
        (Strategies.DP_REPLICATE, Strategies.TP, Strategies.DP_SHARD),
        (1, 4, 2),
        (8, 1, 4),
    )
    assert config.get_mesh_dims() == (
        (Strategies.TP, Strategies.DP_SHARD),
        (4, 2),
    )


def test_parallel_config_folds_equal_tensor_and_expert_parallel_sizes(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr("parallel.parallel.state.is_dist_initialized", lambda: True)
    monkeypatch.setattr("parallel.parallel.state.dist.get_rank", lambda: 0)
    monkeypatch.setattr("parallel.parallel.state.dist.get_world_size", lambda: 8)

    config = ParallelConfig(_config(tp=4, ep=4, dp_shard=2))

    assert config.model_parallel_size == 4
    assert config.total_size == 8
    assert config.get_mesh_layout() == (
        (Strategies.DP_REPLICATE, Strategies.TP, Strategies.DP_SHARD),
        (1, 4, 2),
        (8, 1, 4),
    )


def test_sequence_parallel_reuses_tensor_parallel_mesh(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr("parallel.parallel.state.is_dist_initialized", lambda: True)
    monkeypatch.setattr("parallel.parallel.state.dist.get_rank", lambda: 0)
    monkeypatch.setattr("parallel.parallel.state.dist.get_world_size", lambda: 8)

    config = ParallelConfig(_config(tp=4, sp=4, ep=4, dp_shard=2))

    assert config.total_size == 8
    assert Strategies.SP not in config.get_mesh_layout()[0]


def test_sequence_parallel_must_equal_tensor_parallel_size(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr("parallel.parallel.state.is_dist_initialized", lambda: True)
    monkeypatch.setattr("parallel.parallel.state.dist.get_rank", lambda: 0)
    monkeypatch.setattr("parallel.parallel.state.dist.get_world_size", lambda: 8)

    with pytest.raises(ValueError, match="sp == 1 or sp == tp"):
        ParallelConfig(_config(tp=4, sp=2, ep=4, dp_shard=2))


def test_parallel_config_rejects_unfolded_tensor_and_expert_sizes(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr("parallel.parallel.state.is_dist_initialized", lambda: True)
    monkeypatch.setattr("parallel.parallel.state.dist.get_rank", lambda: 0)
    monkeypatch.setattr("parallel.parallel.state.dist.get_world_size", lambda: 8)

    with pytest.raises(ValueError, match="tp == ep"):
        ParallelConfig(_config(tp=4, ep=2, dp_shard=2))


def test_parallel_config_builds_tp_adjacent_physical_mesh(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr("parallel.parallel.state.is_dist_initialized", lambda: True)
    monkeypatch.setattr("parallel.parallel.state.dist.get_rank", lambda: 0)
    monkeypatch.setattr("parallel.parallel.state.dist.get_world_size", lambda: 8)
    config = ParallelConfig(_config(tp=4, ep=4, dp_shard=2))

    mesh_dim_names, mesh_shape, mesh_strides = config.get_mesh_layout()
    mesh = config._physical_mesh(mesh_dim_names, mesh_shape, mesh_strides)

    assert mesh.tolist() == [
        [
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
    ]


def test_parallel_config_allows_dense_tensor_parallel_without_expert_parallel(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr("parallel.parallel.state.is_dist_initialized", lambda: True)
    monkeypatch.setattr("parallel.parallel.state.dist.get_rank", lambda: 0)
    monkeypatch.setattr("parallel.parallel.state.dist.get_world_size", lambda: 8)

    config = ParallelConfig(_config(tp=4, dp_shard=2))
    assert config.tp_size == 4
    assert config.ep_size == 1


def test_parallel_config_rejects_unsupported_mesh_dimensions(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr("parallel.parallel.state.is_dist_initialized", lambda: True)
    monkeypatch.setattr("parallel.parallel.state.dist.get_rank", lambda: 0)
    monkeypatch.setattr("parallel.parallel.state.dist.get_world_size", lambda: 2)

    with pytest.raises(NotImplementedError, match="tensor parallel"):
        ParallelConfig(_config(cp=2))


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

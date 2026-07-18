import copy
import os
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from parallel.parallel import dataloader as dataloader_module


class FakeTokenizer:
    bos_token_id = 1
    name_or_path = "fake-tokenizer"
    vocab_size = 256

    def __call__(self, documents, add_special_tokens=False):
        assert not add_special_tokens
        return {
            "input_ids": [
                [2 + (ord(character) % 100) for character in document]
                for document in documents
            ]
        }


def _write_dataset(tmp_path):
    train_path = tmp_path / "shard_00000.parquet"
    validation_path = tmp_path / "shard_00001.parquet"
    pq.write_table(
        pa.table({"text": ["a", "bb", "c", "dd", "e", "ff", "g", "hh"]}),
        train_path,
        row_group_size=4,
    )
    pq.write_table(
        pa.table({"text": ["v", "ww", "x", "yy"]}),
        validation_path,
        row_group_size=4,
    )
    return train_path, validation_path


def _loader(monkeypatch, paths, *, resume=None, batch_size=1):
    monkeypatch.setattr(
        dataloader_module,
        "list_data_files",
        lambda: [str(path) for path in paths],
    )
    return dataloader_module.dist_data_loader(
        FakeTokenizer(),
        batch_size,
        3,
        split="train",
        tokenizer_batch_size=2,
        buffer_size=2,
        device="cpu",
        resume_state_dict=resume,
        pconfig=SimpleNamespace(data_rank=0, data_world_size=1),
    )


def test_dataloader_resume_replays_next_batch_without_per_batch_snapshot(
    tmp_path, monkeypatch
):
    paths = _write_dataset(tmp_path)
    loader = _loader(monkeypatch, paths)

    _, _, live_state = next(loader)
    resume_state = copy.deepcopy(live_state)
    expected_inputs, expected_targets, next_state = next(loader)

    assert live_state["doc_buffer"] is next_state["doc_buffer"]

    resumed = _loader(monkeypatch, paths, resume=resume_state)
    actual_inputs, actual_targets, _ = next(resumed)
    torch.testing.assert_close(actual_inputs, expected_inputs)
    torch.testing.assert_close(actual_targets, expected_targets)


def test_dataloader_resume_rejects_dataset_or_packing_changes(tmp_path, monkeypatch):
    paths = _write_dataset(tmp_path)
    loader = _loader(monkeypatch, paths)
    _, _, state = next(loader)
    state = copy.deepcopy(state)

    changed_config_loader = _loader(monkeypatch, paths, resume=state, batch_size=2)
    with pytest.raises(RuntimeError, match="configuration has changed"):
        next(changed_config_loader)

    train_path = paths[0]
    stat = train_path.stat()
    os.utime(
        train_path,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
    )
    changed_dataset_loader = _loader(monkeypatch, paths, resume=state)
    with pytest.raises(RuntimeError, match="dataset files have changed"):
        next(changed_dataset_loader)

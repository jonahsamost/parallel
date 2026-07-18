import copy
from pathlib import Path

import torch
import pyarrow.parquet as pq

from .dataset import list_data_files
from .state import ParallelConfig


def _split_data_files(split: str) -> list[str]:
    file_paths = list_data_files()
    if not file_paths:
        raise RuntimeError("No data found; run dataset.py first")
    if split == "train" and len(file_paths) < 2:
        raise RuntimeError(
            "Training requires at least two data files because the final file "
            "is reserved for validation"
        )
    selected = file_paths[:-1] if split == "train" else file_paths[-1:]
    return [str(Path(path).resolve()) for path in selected]


def _dataset_identity(file_paths: list[str]) -> list[dict[str, int | str]]:
    identity = []
    for file_path in file_paths:
        stat = Path(file_path).stat()
        identity.append(
            {
                "path": file_path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return identity


def _create_batches(
    split: str,
    resume_state_dict: dict,
    tokenizer_batch_size: int,
    data_rank: int,
    data_world_size: int,
    file_paths: list[str] | None = None,
):
    file_paths = file_paths if file_paths is not None else _split_data_files(split)

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_batch_idx = (
        resume_state_dict.get("batch_idx") if resume_state_dict is not None else None
    )
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(file_paths):
            filepath = file_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                if resume_batch_idx is None:
                    base_idx = resume_rg_idx // data_world_size
                    rg_idx = (base_idx + 1) * data_world_size + data_rank
                else:
                    rg_idx = resume_rg_idx
            else:
                rg_idx = data_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                start_batch_idx = 0
                if (
                    first_pass
                    and pq_idx == resume_pq_idx
                    and rg_idx == resume_rg_idx
                    and resume_batch_idx is not None
                ):
                    start_batch_idx = resume_batch_idx + 1
                for batch_idx, i in enumerate(
                    range(0, len(batch), tokenizer_batch_size)
                ):
                    if batch_idx < start_batch_idx:
                        continue
                    yield batch[i: i + tokenizer_batch_size], (
                        pq_idx,
                        rg_idx,
                        epoch,
                        batch_idx,
                    )
                resume_rg_idx = None
                resume_batch_idx = None
                rg_idx += data_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def dist_data_loader(
    tokenizer, B, T, split,
    tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None, buffer_size=1000,
    pconfig: ParallelConfig | None = None,
):
    assert split in ["train", "val"], "split must be train/val"

    row_capacity = T + 1
    data_rank = pconfig.data_rank if pconfig is not None else 0
    data_world_size = pconfig.data_world_size if pconfig is not None else 1
    file_paths = _split_data_files(split)
    dataset_identity = _dataset_identity(file_paths)
    loader_config = {
        "split": split,
        "batch_size": B,
        "sequence_length": T,
        "tokenizer_batch_size": tokenizer_batch_size,
        "buffer_size": buffer_size,
        "data_rank": data_rank,
        "data_world_size": data_world_size,
        "tokenizer_class": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        ),
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
        "bos_token_id": tokenizer.bos_token_id,
    }
    if resume_state_dict is not None:
        if resume_state_dict.get("dataset") != dataset_identity:
            raise RuntimeError(
                "Cannot resume dataloader because the dataset files have changed"
            )
        if resume_state_dict.get("loader_config") != loader_config:
            raise RuntimeError(
                "Cannot resume dataloader because its configuration has changed"
            )
    batches = _create_batches(
        split,
        resume_state_dict,
        tokenizer_batch_size,
        data_rank,
        data_world_size,
        file_paths,
    )
    bos_token_id = tokenizer.bos_token_id
    doc_buffer = copy.deepcopy(
        resume_state_dict.get("doc_buffer", [])
        if resume_state_dict is not None
        else []
    )
    pq_idx, rg_idx, epoch, batch_idx = 0, 0, 1, -1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch, batch_idx
        doc_batch, (pq_idx, rg_idx, epoch, batch_idx) = next(batches)
        encoded = tokenizer(doc_batch, add_special_tokens=False)
        for token_ids in encoded["input_ids"]:
            if bos_token_id is not None:
                token_ids = [bos_token_id] + token_ids
            doc_buffer.append(token_ids)
    
    use_cuda = torch.device(device).type == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos: pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        state_dict = {
            "pq_idx": pq_idx,
            "rg_idx": rg_idx,
            "epoch": epoch,
            "batch_idx": batch_idx,
            # This is a live reference. CheckpointManager snapshots it only when
            # a checkpoint is actually written, avoiding a full copy per batch.
            "doc_buffer": doc_buffer,
            "dataset": dataset_identity,
            "loader_config": loader_config,
        }
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

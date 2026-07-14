import torch
import pyarrow.parquet as pq

from parallel.parallel.state import RuntimeState

def _create_batches(
    split: str,
    resume_state_dict: dict,
    tokenizer_batch_size: int,
):
    state = RuntimeState()
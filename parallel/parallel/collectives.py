import torch

from .state import RuntimeState

def wait_for_everyone(state: RuntimeState):
    if state.backend == "nccl":
        torch.distributed.barrier(device_ids=[state.local_rank])
    else:
        torch.distributed.barrier()
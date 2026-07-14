from parallel.parallel.state import RuntimeState
import torch

def wait_for_everyone(state: RuntimeState):
    if state.backend == "cuda":
        torch.distributed.barrier(device_ids=[state.local_rank])
    else:
        torch.distributed.barrier()
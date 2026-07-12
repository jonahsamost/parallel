import os
from functools import partial
import warnings
from typing import Callable, Any, Tuple
from parallel.engine.utils.imports import is_deepspeed_available
import torch

from .utils.dataclasses import DistType
from .utils.environment import DEFAULT_MASTER_PORT, get_cpu_distributed_information, get_debug_mode, get_fork_launched, get_fsdp_offload_params, get_fsdp_state_dict_type, get_int_from_env, get_local_rank, get_master_addr, get_master_port, get_omp_num_threads, get_torch_device, get_use_cpu, get_use_deepspeed, get_use_fsdp

def empty_fn(*args, **kwargs):
    return None

class RuntimeState:
    """
    Singleton for training env and process control.

    Args:
        - cpu (`bool`)
            Whether or not to execute script on CPU
        - kwargs

    """
    _shared_state = {}
    def __init__(
        self,
        cpu: bool = False,
        backend: str | None = None,
        **kwargs,
    ):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self._cpu = cpu
            self.backend = None
            env_device = get_torch_device()
            self.device = torch.device(env_device) if env_device is not None else None
            self.debug = get_debug_mode()
            final_backend, dist_type = self._prepare_backend(cpu, backend)
            if backend is not None and final_backend != backend:
                raise ValueError(f"Chosen backend {backend} could not be found")
            
            self.backend = final_backend
            self.distributed_type = dist_type
            use_deepspeed = False
            dist_info = None
            if not cpu:
                if get_local_rank() != -1:
                    if get_use_deepspeed():
                        if not is_deepspeed_available():
                            raise ImportError("Trying to use DeepSpeed, but cannot find DeepSpeed")
                        from deepspeed import comm as dist        

                        if not dist.is_initialized():
                            if self.backend == "tccl":
                                local_rank = get_local_rank()
                                torch.sdaa.set_device(f"sdaa:{local_rank}")
                            dist.init_distributed(dist_backend=self.backend, auto_mpi_discovery=False, **kwargs)
                        use_deepspeed = True
                    elif (
                        self.distributed_type is not DistType.MULTI_CPU
                        and not torch.distributed.is_initialized()
                    ):
                        if self.backend == "tccl":
                            local_rank = get_local_rank()
                            torch.sdaa.set_device(f"sdaa:{local_rank}")
                        if (
                            self.backend == "nccl"
                            and get_use_fsdp()
                            and (
                                get_fsdp_offload_params()
                                or get_fsdp_state_dict_type() == "FULL_STATE_DICT"
                            )
                        ):
                            self.backend = "cuda:nccl,cpu:gloo"
                        torch.distributed.init_process_group(backend=self.backend, **kwargs)
            
            if self.distributed_type == DistType.MULTI_CPU:
                dist_info = get_cpu_distributed_information()
                os.environ["RANK"] = str(dist_info.rank)
                os.environ["WORLD_SIZE"] = str(dist_info.world_size)
                os.environ["LOCAL_RANK"] = str(dist_info.local_rank)
                os.environ["LOCAL_WORLD_SIZE"] = str(dist_info.local_world_size)
                if not get_master_port():
                    os.environ["MASTER_PORT"] = DEFAULT_MASTER_PORT
                if (
                    not get_master_addr()
                    and dist_info.local_world_size != dist_info.world_size
                    and self.backend != "mpi"
                ):
                    raise ValueError("MASTER_ADDR not set. Ensure set for participating ranks")
                
                kwargs["rank"] = dist_info.rank
                kwargs["world_size"] = dist_info.world_size

                if (
                    self.distributed_type == DistType.MULTI_CPU
                    and get_omp_num_threads() == 0
                ):
                    import psutil
                    num_cpu_threads_per_proc = int(
                        psutil.cpu_count(logical=False) / dist_info.local_world_size
                    )
                    if num_cpu_threads_per_proc == 0:
                        num_cpu_threads_per_proc = 1
                    torch.set_num_threads(num_cpu_threads_per_proc)
                    warnings.warn(f"OMP_NUM_THREADS was unset. Set to {num_cpu_threads_per_proc}")
                
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend=self.backend, **kwargs)

            if self.backend is None:
                self.distributed_type = DistType.NONE
                self.num_processes = 1
                self.process_idx = 0
                self.local_process_idx = 0
            else:
                self.num_processes = torch.distributed.get_world_size()
                self.process_idx = torch.distributed.get_rank()
                self.local_process_idx = (
                    get_local_rank() if dist_info is None else dist_info.local_rank
                )
            
            self.set_device()
            if use_deepspeed:
                self.distributed_type = DistType.DEEPSPEED
        
        self.fork_launched = get_fork_launched()
    
    def __repr__(self) -> str:
        return (
            f"Distributed environment: {self.distributed_type}{('  Backend: ' + self.backend) if self.backend else ''}\n"
            f"Num processes: {self.num_processes}\n"
            f"Process index: {self.process_idx}\n"
            f"Local process index: {self.local_process_idx}\n"
            f"Device: {self.device}\n"
        )
    
    @property
    def default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def on_main_process(self, function: Callable[..., Any] | None = None):
        """Decorator only running on main process"""
        if not self.initialized:
            raise ValueError("RuntimeState must be initialized before calling this function")
        if self.is_main_process or not self.use_distributed:
            return function
        return empty_fn
    
    def on_local_main_process(self, function: Callable[..., Any] | None = None):
        """Decorator only running on local main process"""
        if not self.initialized:
            raise ValueError("RuntimeState must be initialized before calling this function")
        if self.is_local_main_process or not self.use_distributed:
            return function
        return empty_fn
    
    def on_process(self, function: Callable[..., Any] | None = None, process_idx: int | None = None):
        """Decorator only running on process with given idx"""
        if function is None:
            return partial(self.on_process, process_idx=process_idx)
        if (self.process_idx == process_idx) or (not self.use_distributed):
            return function
        return empty_fn
    
    def _prepare_backend(
        self, cpu: bool = False, backend: str | None = None,
    ) -> Tuple[str, DistType]:
        dist_type = None
        if get_local_rank() != -1 and not cpu:
            if torch.cuda.is_available():
                if backend is None:
                    backend = "nccl"
                dist_type = DistType.MULTI_GPU
        if (
            dist_type is None
            and cpu
            and (
                get_local_rank() != -1
                or get_int_from_env(["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "WORLD_SIZE"], 1) > 1
            )
        ):
            dist_type = DistType.MULTI_CPU
            if backend in (None, "mpi") and torch.distributed.is_mpi_available():
                backend = "mpi"
            else:
                backend = "gloo"
        
        if dist_type is None:
            dist_type = DistType.NONE
        
        return backend, dist_type
    
    def wait_for_everyone(self):
        if not self.use_distributed or not torch.distributed.is_initialized():
            return
        if self.distributed_type in (
            DistType.MULTI_GPU,
            DistType.MULTI_CPU,
            DistType.DEEPSPEED,
            DistType.FSDP
        ):
            torch.distributed.barrier(device_ids=[self.local_process_idx])
    
    def set_device(self):
        if self.device is not None:
            return
        if self.distributed_type == DistType.NONE:
            self.device = torch.device("cpu") if self._cpu else self.default_device 
            return
        device = str(self.distributed_type).split(".")[-1].replace("MULTI_", "").lower()
        if device not in ("cpu", "gpu"):
            raise ValueError(f"Cannot set device for {self.distributed_type} ({device})")
        
        if device == "gpu":
            device = "cuda"
        device_module = getattr(torch, device)
        device_idx = self.local_process_idx % device_module.device_count()
        self.device = torch.device(device, device_idx)
        device_module.set_device(self.device)
    
    def destroy_process_group(self, group=None):
        if self.fork_launched and group is None:
            return
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group(group)

    @staticmethod
    def _reset_state():
        RuntimeState._shared_state.clear()
    
    @property
    def initialized(self) -> bool:
        return self._shared_state != {}
    
    @property
    def use_distributed(self):
        return self.distributed_type != DistType.NONE and self.num_processes > 1
    
    @property
    def is_main_process(self):
        return self.process_idx == 0
    
    @property
    def is_local_main_process(self):
        return self.local_process_idx == 0
    

class AcceleratorState:
    """
    Singleton holding info about current training env
    """
    _shared_state = {}

    def __init__(
        self,
        mixed_precision: str | None = None,
        cpu: bool = False,
        deepspeed_plugin=None,
        fsdp_plugin=None,
        megatron_lm_plugin=None,
        parallelism_config=None,
        **kwargs,
    ):
        self.__dict__ = self._shared_state
        if get_use_cpu():
            cpu = True
        if RuntimeState._shared_state == {}:
            RuntimeState(cpu, **kwargs)
        self.__dict__.update(RuntimeState._shared_state)
        ...


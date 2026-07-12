import os
import weakref
from functools import partial
import warnings
from typing import Callable, Any, Tuple
from parallel.engine.utils.imports import is_deepspeed_available, is_fp8_available
import torch

from .utils.dataclasses import DistType, GradientAccumulationPlugin
from .utils.environment import DEFAULT_MASTER_PORT, get_allow_cp_standalone, get_cpu_distributed_information, get_debug_mode, get_fork_launched, get_fsdp_offload_params, get_fsdp_state_dict_type, get_int_from_env, get_local_rank, get_master_addr, get_master_port, get_mixed_precision, get_omp_num_threads, get_torch_device, get_use_cpu, get_use_deepspeed, get_use_fsdp

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
    
    def print(self, *args, **kwargs):
        if self.is_local_main_process:
            print(*args, **kwargs)
    

class EngineState:
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
        torch_tp_plugin=None,
        parallelism_config=None,
        flam_plugin=None,
        _from_engine: bool = False,
        **kwargs,
    ):
        self.__dict__ = self._shared_state
        if get_use_cpu():
            cpu = True
        if RuntimeState._shared_state == {}:
            RuntimeState(cpu, **kwargs)
        self.__dict__.update(RuntimeState._shared_state)
        self._check_initialized(mixed_precision, cpu)
        if not self.initialized:
            if not _from_engine:
                raise ValueError(
                    "Initialize via `Engine()` before using EngineState directly."
                )
            self.deepspeed_plugins = None
            self.fsdp_plugin = None
            self.flam_plugin = None
            self.torch_tp_plugin = torch_tp_plugin
            self.parallelism_config = parallelism_config
            self.device_mesh = None
            mixed_precision = (
                get_mixed_precision() if mixed_precision is None else mixed_precision.lower()
            )
            if mixed_precision == "fp8":
                if not is_fp8_available():
                    raise ValueError("Using `fp8` requires `torchao`")
                # TODO add check to ensure A100, h100, b200
            if self.distributed_type == DistType.DEEPSPEED and mixed_precision != "fp8":
                self._mixed_precision = "no"
            else:
                self._mixed_precision = mixed_precision
            if get_use_deepspeed() and not cpu:
                if deepspeed_plugin is None:
                    raise ValueError(
                        "DeepSpeed is enabled but no `deepspeed_plugin` was provided. "
                        "Pass a DeepSpeedPlugin when constructing Engine()."
                    )
                self.distributed_type = DistType.DEEPSPEED
                if not isinstance(deepspeed_plugin, dict):
                    deepspeed_plugin.set_mixed_precision(mixed_precision)
                    deepspeed_plugin.select(_from_accelerator_state=True)
                else:
                    for plugin in deepspeed_plugin.values():
                        plugin.set_mixed_precision(mixed_precision)
                    first_plugin = next(iter(deepspeed_plugin.values())) # get first
                    first_plugin.select(_from_accelerator_state=True)
                self.deepspeed_plugins = deepspeed_plugin
            elif flam_plugin is not None:
                self.distributed_type = DistType.FLAM
                self.flam_plugin = flam_plugin
            elif self.distributed_type in [
                DistType.MULTI_GPU,
            ]:
                if not get_allow_cp_standalone():
                    if self.parallelism_config and self.parallelism_config.cp_enabled and fsdp_plugin is None:
                        raise ValueError(
                            "`cp_size > 1` specified in `paralellism_config` but no `fsdp_plugin` provided"
                        )
                    if (
                        self.parallelism_config is not None
                        and self.parallelism_config.cp_enabled
                        and fsdp_plugin is not None
                        and fsdp_plugin.fsdp_version == 1
                    ):
                        raise ValueError("Using `cp_size > 1` requires fsdp2")
                if (get_use_fsdp() or fsdp_plugin is not None) or (
                    self.parallelism_config is not None and self.parallelism_config.cp_enabled
                ):
                    self.distributed_type = DistType.FSDP
                    if self._mixed_precision != "no" and fsdp_plugin is not None:
                        fsdp_plugin.set_mixed_precision(self._mixed_precision)
                    self.fsdp_plugin = fsdp_plugin
            RuntimeState._shared_state["distributed_type"] = self.distributed_type
    
    @property
    def initialized(self) -> bool:
        return self._shared_state != RuntimeState._shared_state
    
    def _check_initialized(self, mixed_precision=None, cpu=None):
        if self.initialized:
            if cpu and self.device.type != "cpu":
                raise ValueError("Cannot reinitialize EngineState with different cpu")
            if (
                mixed_precision is not None
                and mixed_precision != self._mixed_precision
                and self.distributed_type != DistType.DEEPSPEED
            ):
                raise ValueError("Cannot reinitialize EngineState with different mixed precision")
    
    def print(self, *args, **kwargs):
        RuntimeState().print(*args, **kwargs)
    
    def wait_for_everyone(self):
        RuntimeState().wait_for_everyone()

    @staticmethod
    def _reset_state(reset_partial_state: bool = False):
        EngineState._shared_state.clear()
        if reset_partial_state:
            RuntimeState._reset_state()

    def destroy_process_group(self, group=None):
        RuntimeState().destroy_process_group(group)

    @property
    def fork_launched(self):
        return RuntimeState().fork_launched

    @property
    def use_distributed(self):
        return RuntimeState().use_distributed

    @property
    def is_fsdp2(self) -> bool:
        return (
            self.distributed_type == DistType.FSDP
            and self.fsdp_plugin is not None
            and self.fsdp_plugin.fsdp_version == 2
        )

    @property
    def is_main_process(self) -> bool:
        return RuntimeState().is_main_process

    @property
    def is_local_main_process(self) -> bool:
        return RuntimeState().is_local_main_process
    
    @property
    def mixed_precision(self):
        if self.distributed_type == DistType.DEEPSPEED and self._mixed_precision != "fp8":
            config = self.deepspeed_plugin.deepspeed_config
            if config.get("fp16", {}).get("enabled", False):
                mixed_precision = "fp16"
            elif config.get("bf16", {}).get("enabled", False):
                mixed_precision = "bf16"
            else:
                mixed_precision = "no"
        else:
            mixed_precision = self._mixed_precision
        return mixed_precision
        
    @property
    def deepspeed_plugin(self):
        if self.distributed_type != DistType.DEEPSPEED:
            return None
        from .utils.deepspeed import get_active_deepspeed_plugin
        return get_active_deepspeed_plugin(self)
        
    def get_deepspeed_plugin(self, name: str):
        if self.distributed_type != DistType.DEEPSPEED:
            return None
        if not isinstance(self.deepspeed_plugins, dict):
            return self.deepspeed_plugins
        return self.deepspeed_plugins[name]

    def select_deepspeed_plugin(self, name: str):
        if self.distributed_type != DistType.DEEPSPEED:
            return None
        if not isinstance(self.deepspeed_plugins, dict):
            self.deepspeed_plugins.select(_from_accelerator_state=True)
            return
        for key, plugin in self.deepspeed_plugins.items():
            if key != name:
                plugin._unselect()
        self.deepspeed_plugins[name].select(_from_accelerator_state=True)


class GradientState:
    _shared_state = {}

    def __init__(self, gradient_accumulation_plugin: GradientAccumulationPlugin | None = None):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self.sync_gradients = True
            self._dataloader_references_ref = [None]
            self.plugin_kwargs = (
                gradient_accumulation_plugin.to_kwargs() if gradient_accumulation_plugin is not None else {}
            )
            self._is_xla_gradients_synced = False

        if gradient_accumulation_plugin is not None and self.plugin_kwargs != gradient_accumulation_plugin.to_kwargs():
            self.plugin_kwargs = gradient_accumulation_plugin.to_kwargs()

    @property
    def num_steps(self) -> int:
        return self.plugin_kwargs.get("num_steps", 1)

    @property
    def adjust_scheduler(self) -> bool:
        return self.plugin_kwargs.get("adjust_scheduler", False)

    @property
    def sync_with_dataloader(self) -> bool:
        return self.plugin_kwargs.get("sync_with_dataloader", True)

    @property
    def initialized(self) -> bool:
        return GradientState._shared_state != {}

    @property
    def end_of_dataloader(self) -> bool:
        if not self.in_dataloader:
            return False
        return self.active_dataloader.end_of_dataloader

    @property
    def remainder(self) -> int:
        if not self.in_dataloader:
            return -1
        return self.active_dataloader.remainder

    def __repr__(self):
        return (
            f"Sync Gradients: {self.sync_gradients}\n"
            f"At end of current dataloader: {self.end_of_dataloader}\n"
            f"Extra samples added: {self.remainder}\n"
            f"Gradient accumulation plugin: {self.plugin_kwargs}\n"
        )

    def _set_sync_gradients(self, sync_gradients):
        "Private function that sets whether gradients should be synchronized. Users should not have to call this."
        self.sync_gradients = sync_gradients

    def _add_dataloader(self, dataloader):
        self.dataloader_references += [dataloader]

    def _remove_dataloader(self, dataloader):
        self.dataloader_references = [
            dataloader_ref for dataloader_ref in self.dataloader_references if dataloader_ref != dataloader
        ]

    @property
    def active_dataloader(self):
        return self.dataloader_references[-1]

    @property
    def dataloader_references(self):
        return [reference() if reference is not None else reference for reference in self._dataloader_references_ref]

    @dataloader_references.setter
    def dataloader_references(self, references):
        self._dataloader_references_ref = [
            weakref.ref(dataloader) if dataloader is not None else dataloader for dataloader in references
        ]

    @property
    def in_dataloader(self) -> bool:
        return self.active_dataloader is not None

    @staticmethod
    def _reset_state():
        GradientState._shared_state.clear()

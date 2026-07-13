import math
import importlib
from contextlib import suppress
from typing import Optional, Union, Callable
from parallel.engine.logging import get_logger
from parallel.engine.utils.ops import (
    broadcast,
    broadcast_object_list,
    concatenate,
    find_batch_size,
    get_data_structure,
    initialize_tensors,
    send_to_device,
    slice_tensors,
)
import torch
from torch.utils.data import DataLoader, IterableDataset, BatchSampler, RandomSampler

from parallel.engine.state import GradientState, RuntimeState
from parallel.engine.utils.dataclasses import DistType, RNGType
from parallel.engine.utils.random import synchronize_rng_states
from parallel.engine.utils.imports import is_datasets_available, is_torchdata_available


_PYTORCH_DATALOADER_KWARGS = {
    "batch_size": 1,
    "shuffle": False,
    "sampler": None,
    "batch_sampler": None,
    "num_workers": 0,
    "collate_fn": None,
    "pin_memory": False,
    "drop_last": False,
    "timeout": 0,
    "worker_init_fn": None,
    "multiprocessing_context": None,
    "generator": None,
    "prefetch_factor": 2,
    "persistent_workers": False,
    "pin_memory_device": "",
    "in_order": True,
}

logger = get_logger(__name__)


class DataLoaderStateMixin:
    def __init_subclass__(cls, **kwargs):
        cls.end_of_dataloader = False
        cls.remainder = -1

    def reset(self):
        self.end_of_dataloader = False
        self.remainder = -1

    def begin(self):
        self.reset()
        with suppress(Exception):
            if not self._drop_last:
                length = getattr(self.dataset, "total_dataset_length", len(self.dataset))
                self.remainder = length % self.total_batch_size
        self.gradient_state._add_dataloader(self)

    def end(self):
        "Cleans up the gradient state after exiting the dataloader"
        self.gradient_state._remove_dataloader(self)


class DataLoaderAdapter:
    def __init__(self, dataset, use_stateful_dataloader=False, batch_sampler=None, **kwargs):
        self.use_stateful_dataloader = use_stateful_dataloader
        if is_torchdata_available():
            from torchdata.stateful_dataloader import StatefulDataLoader

        if use_stateful_dataloader and not is_torchdata_available():
            raise ImportError(
                "StatefulDataLoader is not available. Please install torchdata version 0.8.0 or higher to use it."
            )
        if use_stateful_dataloader:
            self.base_dataloader = StatefulDataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
        else:
            self.base_dataloader = DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)

        if hasattr(self.base_dataloader, "state_dict"):
            self.dl_state_dict = self.base_dataloader.state_dict()

    def __getattr__(self, name):
        if name == "base_dataloader":
            raise AttributeError()
        return getattr(self.base_dataloader, name)

    def state_dict(self):
        return self.dl_state_dict

    def load_state_dict(self, state_dict):
        self.base_dataloader.load_state_dict(state_dict)

    @property
    def __class__(self):
        return self.base_dataloader.__class__

    def __len__(self):
        return len(self.base_dataloader)

    def adjust_state_dict_for_prefetch(self):
        if RuntimeState().distributed_type != DistType.NONE:
            factor = RuntimeState().num_processes - 1
            if "_sampler_iter_yielded" in self.dl_state_dict and self.dl_state_dict["_sampler_iter_yielded"] > 0:
                self.dl_state_dict["_sampler_iter_yielded"] -= factor
            if "_num_yielded" in self.dl_state_dict and self.dl_state_dict["_num_yielded"] > 0:
                self.dl_state_dict["_num_yielded"] -= factor
            if self.dl_state_dict.get("_index_sampler_state") is not None:
                if (
                    "samples_yielded" in self.dl_state_dict["_index_sampler_state"]
                    and self.dl_state_dict["_index_sampler_state"]["samples_yielded"] > 0
                ):
                    self.dl_state_dict["_index_sampler_state"]["samples_yielded"] -= self.batch_size * factor

    def _update_state_dict(self):
        if hasattr(self.base_dataloader, "state_dict"):
            self.dl_state_dict = self.base_dataloader.state_dict()
            self.adjust_state_dict_for_prefetch()
            self.dl_state_dict["_iterator_finished"] = self.end_of_dataloader


class DataLoaderShard(DataLoaderAdapter, DataLoaderStateMixin):
    """
    Subclass of `DataLoaderAdapter` that will deal with device placement and current distributed setup.
    """

    def __init__(
        self,
        dataset,
        device=None,
        rng_types=None,
        synchronized_generator=None,
        skip_batches=0,
        use_stateful_dataloader=False,
        _drop_last: bool = False,
        _non_blocking: bool = False,
        torch_device_mesh=None,
        iteration=0,
        **kwargs,
    ):
        super().__init__(dataset, use_stateful_dataloader=use_stateful_dataloader, **kwargs)
        self.device = device
        self.rng_types = rng_types
        self.synchronized_generator = synchronized_generator
        self.skip_batches = skip_batches
        self.gradient_state = GradientState()
        self._drop_last = _drop_last
        self._non_blocking = _non_blocking
        self.iteration = iteration

    def adjust_state_dict_for_prefetch(self):
        # DataLoaderShard does not need the DDP prefetch adjustment that DataLoaderDispatcher needs.
        # In DataLoaderShard, each process has its own sharded base dataloader and the 1-batch
        # look-ahead is already accounted for by the timing of _update_state_dict() calls
        # (called before the inner next(), so the captured state already equals the number of
        # batches yielded to the user).
        pass

    def __iter__(self):
        if self.rng_types is not None:
            synchronize_rng_states(self.rng_types, self.synchronized_generator)
        self.begin()

        self.set_epoch(self.iteration)
        dataloader_iter = self.base_dataloader.__iter__()
        # We iterate one batch ahead to check when we are at the end
        try:
            current_batch = next(dataloader_iter)
        except StopIteration:
            self.end()
            return

        batch_index = 0
        while True:
            try:
                # But we still move it to the device so it is done before `StopIteration` is reached
                if self.device is not None:
                    current_batch = send_to_device(current_batch, self.device, non_blocking=self._non_blocking)
                self._update_state_dict()
                next_batch = next(dataloader_iter)
                if batch_index >= self.skip_batches:
                    yield current_batch
                batch_index += 1
                current_batch = next_batch
            except StopIteration:
                self.end_of_dataloader = True
                self._update_state_dict()
                if batch_index >= self.skip_batches:
                    yield current_batch
                break

        self.iteration += 1
        self.end()

    def __reduce__(self):
        """
        Define the `__reduce__` method to ensure a `DataLoaderShard` can be pickled and unpickled. This needs to be
        explicitly defined since default pickling behavior is broken by `DataLoaderAdapter` messing with its
        `__class__` member.
        """
        args = super().__reduce__()
        return (DataLoaderShard, *args[1:])

    def set_epoch(self, epoch: int):
        # In case it is manually passed in, the user can set it to what they like
        if self.iteration != epoch:
            self.iteration = epoch
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)
        if hasattr(self.batch_sampler, "sampler") and hasattr(self.batch_sampler.sampler, "set_epoch"):
            self.batch_sampler.sampler.set_epoch(epoch)
        if (
            hasattr(self.batch_sampler, "batch_sampler")
            and hasattr(self.batch_sampler.batch_sampler, "sampler")
            and hasattr(self.batch_sampler.batch_sampler.sampler, "set_epoch")
        ):
            self.batch_sampler.batch_sampler.sampler.set_epoch(epoch)
        # We support if a custom `Dataset` implementation has `set_epoch`
        # or in general HF datasets `Datasets`
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    @property
    def total_batch_size(self):
        batch_sampler = self.sampler if isinstance(self.sampler, BatchSampler) else self.batch_sampler
        return (
            batch_sampler.batch_size
            if getattr(batch_sampler, "split_batches", False)
            else (batch_sampler.batch_size * getattr(batch_sampler, "num_processes", 1))
        )

    @property
    def total_dataset_length(self):
        if hasattr(self.dataset, "total_length"):
            return self.dataset.total_length
        else:
            return len(self.dataset)

    def get_sampler(self):
        return get_sampler(self)

    def set_sampler(self, sampler):
        sampler_is_batch_sampler = isinstance(self.sampler, BatchSampler)
        if sampler_is_batch_sampler:
            self.sampler.sampler = sampler
        else:
            self.batch_sampler.sampler = sampler
            if hasattr(self.batch_sampler, "batch_sampler"):
                self.batch_sampler.batch_sampler.sampler = sampler


class DataLoaderDispatcher(DataLoaderAdapter, DataLoaderStateMixin):
    """
    Subclass of `DataLoaderAdapter` that will iterate and preprocess on process 0 only, then dispatch on each process
    their part of the batch.
    """

    def __init__(
        self,
        dataset,
        split_batches: bool = False,
        skip_batches=0,
        use_stateful_dataloader=False,
        _drop_last: bool = False,
        _non_blocking: bool = False,
        slice_fn=None,
        torch_device_mesh=None,
        iteration=0,
        **kwargs,
    ):
        shuffle = False
        from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

        # We need to save the shuffling state of the DataPipe
        if isinstance(dataset, ShufflerIterDataPipe):
            shuffle = dataset._shuffle_enabled
        super().__init__(dataset, use_stateful_dataloader=use_stateful_dataloader, **kwargs)
        self.split_batches = split_batches
        if shuffle:
            torch.utils.data.graph_settings.apply_shuffle_settings(dataset, shuffle=shuffle)

        self.gradient_state = GradientState()
        self.state = RuntimeState()
        self._drop_last = _drop_last
        self._non_blocking = _non_blocking
        self.skip_batches = skip_batches
        self.torch_device_mesh = torch_device_mesh

        self.slice_fn = slice_tensors if slice_fn is None else slice_fn
        self.iteration = iteration

        # if a device mesh is provided extract each dimension (dp, fsdp, tp)
        # device mesh may hold any number of dimensions, however,
        # below code is for targeted support for dp, fsdp and tp

        # device mesh will be used only if there is tp involved
        # or any multi-dimensional parallelism involving tp
        # (dp, tp) (fsdp, tp) (dp, fsdp, tp)
        # otherwise the default behaviour not using device mesh should be sufficient
        # since multi dimensional parallelism devoid of tp would anyway need
        # different batches for each process irrespective of dp or fsdp
        self.submesh_tp = None
        self.submesh_dp = None
        self.submesh_fsdp = None
        if self.torch_device_mesh and "tp" in self.torch_device_mesh.mesh_dim_names:
            self.submesh_tp = self.torch_device_mesh["tp"]
            if "dp" in self.torch_device_mesh.mesh_dim_names:
                self.submesh_dp = self.torch_device_mesh["dp"]
            if "fsdp" in self.torch_device_mesh.mesh_dim_names:
                self.submesh_fsdp = self.torch_device_mesh["fsdp"]
        if self.submesh_tp and (self.submesh_dp or self.submesh_fsdp):
            raise ValueError("TP + (DP/FSDP) is not yet supported in dispatch mode")

    def _fetch_batches(self, iterator):
        batches, batch = None, None
        # On process 0, we gather the batch to dispatch.
        if self.state.process_idx == 0:
            # Procedure to support TP only is simpler
            # since we want to dispatch the same batch of samples across all ranks
            # this removes complexity of handling multiple tp rank groups when TP + DP
            # combination is involved.

            try:
                # for TP case avoid using split_batches
                # since it would mean that the dataloader should be spilling out
                # duplicates of batches.
                if self.split_batches:
                    # One batch of the main iterator is dispatched and split.
                    if self.submesh_tp:
                        logger.warning(
                            "Use of split_batches for TP would need the dataloader to produce duplicate batches,"
                            "otherwise, use dispatch_batches=True instead."
                        )
                    self._update_state_dict()
                    batch = next(iterator)
                else:
                    # num_processes batches of the main iterator are concatenated then dispatched and split.
                    # We add the batches one by one so we have the remainder available when drop_last=False.
                    batches = []
                    if self.submesh_tp:
                        # when tp, extract single batch and then replicate
                        self._update_state_dict()
                        batch = next(iterator)
                        batches = [batch] * self.state.num_processes
                    else:
                        for _ in range(self.state.num_processes):
                            self._update_state_dict()
                            batches.append(next(iterator))
                    try:
                        batch = concatenate(batches, dim=0)
                    except RuntimeError as e:
                        raise RuntimeError(
                            "You can't use batches of different size with `dispatch_batches=True` or when using an `IterableDataset`."
                            "either pass `dispatch_batches=False` and have each process fetch its own batch "
                            " or pass `split_batches=True`. By doing so, the main process will fetch a full batch and "
                            "slice it into `num_processes` batches for each process."
                        ) from e
                # In both cases, we need to get the structure of the batch that we will broadcast on other
                # processes to initialize the tensors with the right shape.
                # data_structure, stop_iteration
                batch_info = [get_data_structure(batch), False]
            except StopIteration:
                batch_info = [None, True]
        else:
            batch_info = [None, self._stop_iteration]
        # This is inplace, so after this instruction, every process has the same `batch_info` as process 0.
        broadcast_object_list(batch_info)
        self._stop_iteration = batch_info[1]
        if self._stop_iteration:
            # If drop_last is False and split_batches is False, we may have a remainder to take care of.
            if not self.split_batches and not self._drop_last:
                if self.state.process_idx == 0 and len(batches) > 0:
                    batch = concatenate(batches, dim=0)
                    batch_info = [get_data_structure(batch), False]
                else:
                    batch_info = [None, True]
                broadcast_object_list(batch_info)
        return batch, batch_info

    def __iter__(self):
        self.begin()
        self.set_epoch(self.iteration)
        main_iterator = self.base_dataloader.__iter__()
        stop_iteration = False
        self._stop_iteration = False
        first_batch = None
        next_batch, next_batch_info = self._fetch_batches(main_iterator)
        batch_index = 0
        while not stop_iteration:
            batch, batch_info = next_batch, next_batch_info

            if self.state.process_idx != 0:
                # Initialize tensors on other processes than process 0.
                batch = initialize_tensors(batch_info[0])
            batch = send_to_device(batch, self.state.device, non_blocking=self._non_blocking)
            # Broadcast the batch before splitting it.
            batch = broadcast(batch, from_process=0)

            if not self._drop_last and first_batch is None:
                # We keep at least num processes elements of the first batch to be able to complete the last batch
                first_batch = self.slice_fn(
                    batch,
                    slice(0, self.state.num_processes),
                    process_index=self.state.process_idx,
                    num_processes=self.state.num_processes,
                )

            if batch is None:
                raise ValueError(
                    f"Batch does not contain any data (`{batch}`). At the end of all iterable data available before expected stop iteration."
                )

            observed_batch_size = find_batch_size(batch)
            batch_size = observed_batch_size // self.state.num_processes

            stop_iteration = self._stop_iteration
            if not stop_iteration:
                # We may still be at the end of the dataloader without knowing it yet: if there is nothing left in
                # the dataloader since the number of batches is a round multiple of the number of processes.
                next_batch, next_batch_info = self._fetch_batches(main_iterator)
                # next_batch_info[0] is None when there are no more batches, otherwise we still need to process them.
                if self._stop_iteration and next_batch_info[0] is None:
                    stop_iteration = True

            if not self._drop_last and stop_iteration and observed_batch_size % self.state.num_processes != 0:
                # If the last batch is not complete, let's add the first batch to it.
                batch = concatenate([batch, first_batch], dim=0)
                # Batch size computation above is wrong, it's off by 1 so we fix it.
                batch_size += 1

            data_slice = slice(self.state.process_idx * batch_size, (self.state.process_idx + 1) * batch_size)
            batch = self.slice_fn(
                batch,
                data_slice,
                process_index=self.state.process_idx,
                num_processes=self.state.num_processes,
            )

            if stop_iteration:
                self.end_of_dataloader = True
                self._update_state_dict()
                self.remainder = observed_batch_size
            if batch_index >= self.skip_batches:
                yield batch
            batch_index += 1
        self.iteration += 1
        self.end()

    def set_epoch(self, epoch: int):
        # In case it is manually passed in, the user can set it to what they like
        if self.iteration != epoch:
            self.iteration = epoch
        if hasattr(self.batch_sampler, "sampler") and hasattr(self.batch_sampler.sampler, "set_epoch"):
            self.batch_sampler.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __len__(self):
        whole_length = len(self.base_dataloader)
        if self.split_batches:
            return whole_length
        elif self._drop_last:
            return whole_length // self.state.num_processes
        else:
            return math.ceil(whole_length / self.state.num_processes)

    def __reduce__(self):
        """
        Define the `__reduce__` method to ensure a `DataLoaderDispatcher` can be pickled and unpickled. This needs to
        be explicitly defined since default pickling behavior is broken by `DataLoaderAdapter` messing with its
        `__class__` member.
        """
        args = super().__reduce__()
        return (DataLoaderDispatcher, *args[1:])

    @property
    def total_batch_size(self):
        return (
            self.dataset.batch_size if self.split_batches else (self.dataset.batch_size * self.dataset.num_processes)
        )

    @property
    def total_dataset_length(self):
        return len(self.dataset)

    def get_sampler(self):
        return get_sampler(self)

    def set_sampler(self, sampler):
        sampler_is_batch_sampler = isinstance(self.sampler, BatchSampler)
        if sampler_is_batch_sampler:
            self.sampler.sampler = sampler
        else:
            self.batch_sampler.sampler = sampler
            if hasattr(self.batch_sampler, "batch_sampler"):
                self.batch_sampler.batch_sampler.sampler = sampler


class SeedableRandomSampler(RandomSampler):
    def __init__(self, *args, **kwargs):
        data_seed = kwargs.pop("data_seed", None)
        super().__init__(*args, **kwargs)
        self.initial_seed = data_seed if data_seed is not None else torch.random.initial_seed()
        self.epoch = 0
    
    def __iter__(self):
        if self.generator is None:
            self.generator = torch.Generator(
                device=torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
            )
            self.generator.manual_seed(self.initial_seed)
        
        seed = self.epoch + self.initial_seed
        self.generator.manual_seed(seed)
        yield from super().__iter__()
        self.set_epoch(self.epoch + 1)
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch


class IterableDatasetShard(IterableDataset):
    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
        split_batches: bool = False,
    ):
        if split_batches and batch_size > 1 and batch_size % num_processes != 0:
            raise ValueError(
                f"To use `IterableDatasetShard` in `split_batches` mode, the batch size ({batch_size}) "
                f"needs to be a round multiple of the number of processes ({num_processes})."
            )
        self.dataset: IterableDataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index
        self.split_batches = split_batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __len__(self):
        if self.drop_last:
            return (len(self.dataset) // (self.batch_size * self.num_processes)) * self.batch_size
        else:
            return math.ceil(len(self.dataset) / (self.batch_size * self.num_processes)) * self.batch_size
        
    def __iter__(self):
        if (
            not hasattr(self.dataset, "set_epoch")
            and hasattr(self.dataset, "generator")
            and isinstance(self.dataset.generator, torch.Generator)
        ):
            self.dataset.generator.manual_seed(self.epoch)
        real_batch_size = self.batch_size if self.split_batches else (self.batch_size * self.num_processes)
        process_batch_size = (self.batch_size // self.num_processes) if self.split_batches else self.batch_size
        process_slice = range(self.process_index * process_batch_size, (self.process_index + 1) * process_batch_size)

        first_batch = None
        current_batch = []
        for element in self.dataset:
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield current_batch[i]
                if first_batch is None:
                    first_batch = current_batch.copy()
                current_batch = []

        # Finished if drop_last is True, otherwise complete the last batch with elements from the beginning.
        if not self.drop_last and len(current_batch) > 0:
            if first_batch is None:
                first_batch = current_batch.copy()
            while len(current_batch) < real_batch_size:
                current_batch += first_batch
            for i in process_slice:
                yield current_batch[i]


class BatchSamplerShard(BatchSampler):
    def __init__(
        self,
        batch_sampler: BatchSampler,
        num_processes: int = 1,
        process_index: int = 0,
        split_batches: bool = False,
        even_batches: bool = True,
    ):
        self.batch_sampler = batch_sampler
        self.num_processes = num_processes
        self.process_index = process_index
        self.split_batches = split_batches
        self.even_batches = even_batches
        self.batch_size = getattr(batch_sampler, "batch_size", None)
        self.drop_last = getattr(batch_sampler, "drop_last", False)
        if split_batches and (self.batch_size is None or self.batch_size % num_processes != 0):
            raise ValueError(
                f"To use `BatchSamplerShard` in `split_batches` mode, the batch size ({self.batch_size}) "
                f"needs to be a round multiple of the number of processes ({num_processes})."
            )

    @property
    def total_length(self):
        return len(self.batch_sampler)

    def __len__(self):
        if self.split_batches:
            # Split batches does not change the length of the batch sampler
            return len(self.batch_sampler)
        if len(self.batch_sampler) % self.num_processes == 0:
            # If the length is a round multiple of the number of processes, it's easy.
            return len(self.batch_sampler) // self.num_processes
        length = len(self.batch_sampler) // self.num_processes
        if self.drop_last:
            # Same if we drop the remainder.
            return length
        elif self.even_batches:
            # When we even batches we always get +1
            return length + 1
        else:
            # Otherwise it depends on the process index.
            return length + 1 if self.process_index < len(self.batch_sampler) % self.num_processes else length

    def __iter__(self):
        return self._iter_with_split() if self.split_batches else self._iter_with_no_split()

    def _iter_with_split(self):
        initial_data = []
        batch_length = self.batch_sampler.batch_size // self.num_processes
        for idx, batch in enumerate(self.batch_sampler):
            if idx == 0:
                initial_data = batch
            if len(batch) == self.batch_size:
                # If the batch is full, we yield the part of it this process is responsible of.
                yield batch[batch_length * self.process_index : batch_length * (self.process_index + 1)]

        # If drop_last is True of the last batch was full, iteration is over, otherwise...
        if not self.drop_last and len(initial_data) > 0 and len(batch) < self.batch_size:
            if not self.even_batches:
                if len(batch) > batch_length * self.process_index:
                    yield batch[batch_length * self.process_index : batch_length * (self.process_index + 1)]
            else:
                # For degenerate cases where the dataset has less than num_process * batch_size samples
                while len(initial_data) < self.batch_size:
                    initial_data += initial_data
                batch = batch + initial_data
                yield batch[batch_length * self.process_index : batch_length * (self.process_index + 1)]

    def _iter_with_no_split(self):
        initial_data = []
        batch_to_yield = None
        for idx, batch in enumerate(self.batch_sampler):
            # We gather the initial indices in case we need to circle back at the end.
            if not self.drop_last and idx < self.num_processes:
                if self.batch_size is None:
                    # If batch size is None, `batch` is considered to be a list of indices with dynamic length.
                    initial_data.append(batch)
                else:
                    initial_data += batch
            # We identify the batch to yield but wait until we ar sure every process gets a full batch before actually
            # yielding it.
            if idx % self.num_processes == self.process_index:
                batch_to_yield = batch
            if idx % self.num_processes == self.num_processes - 1 and (
                self.batch_size is None or len(batch) == self.batch_size
            ):
                yield batch_to_yield
                batch_to_yield = None

        # If drop_last is True, iteration is over, otherwise...
        if not self.drop_last and len(initial_data) > 0:
            if not self.even_batches:
                if batch_to_yield:
                    yield batch_to_yield
            else:
                # ... we yield the complete batch we had saved before if it has the proper length
                if batch_to_yield and (self.batch_size is None or len(batch_to_yield) == self.batch_size):
                    yield batch_to_yield

                # For degenerate cases where the dataset has less than num_process * batch_size samples
                _min_length_needed = (
                    self.num_processes * self.batch_size if self.batch_size is not None else self.num_processes
                )
                while len(initial_data) < _min_length_needed:
                    initial_data += initial_data

                # If the last batch seen was of the proper size, it has been yielded by its process so we move to the next
                if self.batch_size is None or len(batch) == self.batch_size:
                    batch = []
                    idx += 1

                # Make sure we yield a multiple of self.num_processes batches
                cycle_index = 0
                while idx % self.num_processes != 0 or len(batch) > 0:
                    if self.batch_size is None:
                        batch = initial_data[cycle_index]
                        if idx % self.num_processes == self.process_index:
                            yield batch
                        cycle_index += 1
                    else:
                        end_index = cycle_index + self.batch_size - len(batch)
                        batch += initial_data[cycle_index:end_index]
                        if idx % self.num_processes == self.process_index:
                            yield batch
                        cycle_index = end_index
                    batch = []
                    idx += 1


def get_sampler(dataloader):
    sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
    if sampler_is_batch_sampler:
        sampler = getattr(dataloader.sampler, "sampler", None)
    else:
        sampler = getattr(dataloader.batch_sampler, "sampler", None)
    return sampler


def prepare_data_loader(
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_processes: Optional[int] = None,
    process_index: Optional[int] = None,
    split_batches: bool = False,
    put_on_device: bool = False,
    rng_types: Optional[list[Union[str, RNGType]]] = None,
    dispatch_batches: Optional[bool] = None,
    even_batches: bool = True,
    slice_fn_for_dispatch: Optional[Callable] = None,
    use_seedable_sampler: bool = False,
    data_seed: Optional[int] = None,
    non_blocking: bool = False,
    use_stateful_dataloader: bool = False,
    torch_device_mesh=None,
) -> DataLoader:
    """
        rng_types (list of `str` or `RNGType`):
            Random number generators to synchronize at the start of each iteration.
            Supported values: `"torch"`, `"cuda"`, `"generator"`.
        dispatch_batches (`bool`, *optional*):
            If set to `True`, the dataloader prepared is only iterated through on the main process and then the batches
            are split and broadcast to each process. Will default to `True` when the underlying dataset is an
            `IterableDataset`, `False` otherwise.
        torch_device_mesh (`torch.distributed.DeviceMesh`, *optional*, defaults to `None`):
            PyTorch device mesh.
    """
    if dispatch_batches is None:
        if not put_on_device:
            dispatch_batches = False
        else:
            dispatch_batches = isinstance(dataloader.dataset, IterableDataset)
    
    if dispatch_batches and not put_on_device:
        raise ValueError("Using `dispatch_batches=True` requires `put_on_device=True`")
    
    state = RuntimeState()
    if num_processes is None:
        num_processes = state.num_processes
    
    if process_index is None:
        process_index = state.process_idx
    
    if torch_device_mesh:
        if state.distributed_type == DistType.DEEPSPEED:
            submesh_tp_size = 1
            if "tp" in torch_device_mesh.mesh_dim_names:
                submesh_tp_size = torch_device_mesh["tp"].size()
            process_index = process_index // submesh_tp_size
            num_processes = num_processes // submesh_tp_size
        else:
            submesh_fsdp_size = 1
            submesh_dp_size = 1
            submesh_tp_size = 1
            submesh_cp_size = 1
            if "tp" in torch_device_mesh.mesh_dim_names:
                submesh_tp_size = torch_device_mesh["tp"].size()
            if "cp" in torch_device_mesh.mesh_dim_names:
                submesh_cp_size = torch_device_mesh["cp"].size()
            if "dp_replicate" in torch_device_mesh.mesh_dim_names:
                submesh_dp_size = torch_device_mesh["dp_replicate"].size()
            if "dp_shard" in torch_device_mesh.mesh_dim_names:
                submesh_fsdp_size = torch_device_mesh["dp_shard"].size()
            process_index = process_index // (submesh_tp_size * submesh_cp_size)
            num_processes = submesh_fsdp_size * submesh_dp_size
    
    if split_batches:
        if dataloader.batch_size is not None:
            batch_size_for_check = dataloader.batch_size
        else:
            if hasattr(dataloader.batch_sampler, "batch_size"):
                batch_size_for_check = dataloader.batch_sampler.batch_size
            else:
                raise ValueError(
                    "To use `split_batches=True` you must have a `batch_size` attribute in the passed `dataloader`"
                )
        if batch_size_for_check > 1 and batch_size_for_check % num_processes != 0:
            raise ValueError(
                f"To use a `DataLoader` in `split_batches` mode, the batch size ({dataloader.batch_size}) "
                f"needs to be a round multiple of the number of processes ({num_processes})."
            )
    
    new_dataset = dataloader.dataset
    new_batch_sampler = dataloader.batch_sampler if not isinstance(new_dataset, IterableDataset) else None
    sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
    synchronized_generator = None

    sampler = get_sampler(dataloader)
    if isinstance(sampler, RandomSampler) and use_seedable_sampler:
        sampler = SeedableRandomSampler(
            data_source=sampler.data_source,
            replacement=sampler.replacement,
            num_samples=sampler._num_samples,
            generator=getattr(
                sampler,
                "generator",
                torch.Generator(device=torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu")
            ),
            data_seed=data_seed,
        )
    
    if num_processes != 1 and not dispatch_batches:
        if is_datasets_available():
            from datasets import IterableDataset as DatasetsIterableDataset
        if (
            is_datasets_available()
            and isinstance(new_dataset, DatasetsIterableDataset)
            and not split_batches
            and new_dataset.n_shards >= num_processes
        ):
            new_dataset = new_dataset.shard(num_shards=num_processes, index=process_index)
        elif isinstance(new_dataset, IterableDataset):
            if getattr(dataloader.dataset, "generator", None) is not None:
                synchronized_generator = dataloader.dataset.generator
            new_dataset = IterableDatasetShard(
                new_dataset,
                batch_size=dataloader.batch_size,
                drop_last=dataloader.drop_last,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )
        else:
            if not use_seedable_sampler and hasattr(sampler, "generator"):
                if sampler.generator is None:
                    sampler.generator = torch.Generator(
                        device=torch.get_default_device() if hasattr(torch, "get_default_device")
                        else "cpu"
                    )
                    seed = int(torch.empty((), dtype=torch.int64).random_().item())
                    sampler.generator.manual_seed(seed)
                synchronized_generator = sampler.generator
            batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
            new_batch_sampler = BatchSamplerShard(
                batch_sampler,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
                even_batches=even_batches,
            )
    
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]

    if rng_types is not None:
        rng_types = list(rng_types)
        if synchronized_generator is None and "generator" in rng_types:
            rng_types.remove("generator")

    kwargs = {
        k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k])
        for k in _PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    if new_batch_sampler is None:
        kwargs["drop_last"] = dataloader.drop_last
        kwargs["batch_size"] = (
            dataloader.batch_size // num_processes if split_batches and not dispatch_batches else dataloader.batch_size
        )

    if dispatch_batches:
        kwargs.pop("generator")
        dataloader = DataLoaderDispatcher(
            new_dataset,
            split_batches=split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader.drop_last,
            _non_blocking=non_blocking,
            slice_fn=slice_fn_for_dispatch,
            use_stateful_dataloader=use_stateful_dataloader,
            torch_device_mesh=torch_device_mesh,
            **kwargs,
        )
    elif sampler_is_batch_sampler:
        dataloader = DataLoaderShard(
            new_dataset,
            device=device if put_on_device else None,
            sampler=new_batch_sampler,
            batch_size=dataloader.batch_size,
            rng_types=rng_types,
            _drop_last=dataloader.drop_last,
            _non_blocking=non_blocking,
            synchronized_generator=synchronized_generator,
            use_stateful_dataloader=use_stateful_dataloader,
            **kwargs,
        )
    else:
        dataloader = DataLoaderShard(
            new_dataset,
            device=device if put_on_device else None,
            batch_sampler=new_batch_sampler,
            rng_types=rng_types,
            synchronized_generator=synchronized_generator,
            _drop_last=dataloader.drop_last,
            _non_blocking=non_blocking,
            use_stateful_dataloader=use_stateful_dataloader,
            **kwargs,
        )

    if isinstance(sampler, SeedableRandomSampler) and use_seedable_sampler:
        dataloader.set_sampler(sampler)
    return dataloader
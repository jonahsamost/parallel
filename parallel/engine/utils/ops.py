from typing import Any
from collections.abc import Mapping
from functools import wraps
from parallel.engine.state import RuntimeState
from parallel.engine.utils.dataclasses import TORCH_DISTRIBUTED_OPERATION_TYPES, DistType, TensorInformation
import torch


def verify_operation(function):
    """
    Verifies that `tensor` is the same shape across all processes. Only ran if `PartialState().debug` is `True`.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        if RuntimeState().distributed_type == DistType.NONE or not RuntimeState().debug:
            return function(*args, **kwargs)
        operation = f"{function.__module__}.{function.__name__}"
        if "tensor" in kwargs:
            tensor = kwargs["tensor"]
        else:
            tensor = args[0]
        if RuntimeState().device.type != find_device(tensor).type:
            raise Exception(
                f"One or more of the tensors passed to {operation} were not on the {tensor.device.type} while the `Accelerator` is configured for {RuntimeState().device.type}. "
                f"Please move it to the {RuntimeState().device.type} before calling {operation}."
            )
        shapes = get_shape(tensor)
        output = gather_object([shapes])
        if output[0] is not None:
            are_same = output.count(output[0]) == len(output)
            if not are_same:
                process_shape_str = "\n  - ".join([f"Process {i}: {shape}" for i, shape in enumerate(output)])
                raise Exception(
                    f"Cannot apply desired operation due to shape mismatches. "
                    "All shapes across devices must be valid."
                    f"\n\nOperation: `{operation}`\nInput shapes:\n  - {process_shape_str}"
                )
        return function(*args, **kwargs)

    return wrapper


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)

def is_namedtuple(data):
    return isinstance(data, tuple) and hasattr(data, "_asdict") and hasattr(data, "_fields")

def honor_type(obj, generator):
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)

def is_tensor_information(tensor_info):
    return isinstance(tensor_info, TensorInformation)

def recursively_apply(func, data, *args, test_type=is_torch_tensor, error_on_other_type=False, **kwargs):
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func, o, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for o in data
            ),
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for k, v in data.items()
            }
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Unsupported types ({type(data)}) passed to `{func.__name__}`. Only nested list/tuple/dicts of "
            f"objects that are valid for `{test_type.__name__}` should be passed."
        )
    return data

def slice_tensors(data, tensor_slice, process_index=None, num_processes=None):
    def _slice_tensor(tensor, tensor_slice):
        return tensor[tensor_slice]

    return recursively_apply(_slice_tensor, data, tensor_slice)


def concatenate(data, dim=0):
    if isinstance(data[0], (tuple, list)):
        return honor_type(data[0], (concatenate([d[i] for d in data], dim=dim) for i in range(len(data[0]))))
    elif isinstance(data[0], Mapping):
        return type(data[0])({k: concatenate([d[k] for d in data], dim=dim) for k in data[0].keys()})
    elif isinstance(data[0], torch.Tensor):
        return torch.cat(data, dim=dim)
    elif isinstance(data, (tuple, list)) and len(data) == 1:
        return data[0]
    else:
        raise TypeError(f"Can only concatenate tensors but got {type(data[0])}")

def get_data_structure(data):
    def _get_data_structure(tensor):
        return TensorInformation(shape=tensor.shape, dtype=tensor.dtype)

    return recursively_apply(_get_data_structure, data)


def broadcast_object_list(object_list, from_process: int = 0):
    if RuntimeState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        torch.distributed.broadcast_object_list(object_list, src=from_process)
    return object_list


def initialize_tensors(data_structure):
    def _initialize_tensor(tensor_info):
        return torch.empty(*tensor_info.shape, dtype=tensor_info.dtype)

    return recursively_apply(_initialize_tensor, data_structure, test_type=is_tensor_information)


def send_to_device(tensor, device, non_blocking=False, skip_keys=None):
    if is_torch_tensor(tensor) or hasattr(tensor, "to"):
        try:
            return tensor.to(device, non_blocking=non_blocking)
        except TypeError:  # .to() doesn't accept non_blocking as kwarg
            return tensor.to(device)
        except AssertionError as error:
            raise error
    elif isinstance(tensor, (tuple, list)):
        return honor_type(
            tensor, (send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys) for t in tensor)
        )
    elif isinstance(tensor, Mapping):
        if isinstance(skip_keys, str):
            skip_keys = [skip_keys]
        elif skip_keys is None:
            skip_keys = []
        return type(tensor)(
            {
                k: t if k in skip_keys else send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys)
                for k, t in tensor.items()
            }
        )
    else:
        return tensor

def find_device(data):
    if isinstance(data, Mapping):
        for obj in data.values():
            device = find_device(obj)
            if device is not None:
                return device
    elif isinstance(data, (tuple, list)):
        for obj in data:
            device = find_device(obj)
            if device is not None:
                return device
    elif isinstance(data, torch.Tensor):
        return data.device


def get_shape(data):
    def _get_shape(tensor):
        return list(tensor.shape)

    return recursively_apply(_get_shape, data)


def find_batch_size(data):
    if isinstance(data, (tuple, list, Mapping)) and (len(data) == 0):
        raise ValueError(f"Cannot find the batch size from empty {type(data)}.")

    if isinstance(data, (tuple, list)):
        return find_batch_size(data[0])
    elif isinstance(data, Mapping):
        for k in data.keys():
            return find_batch_size(data[k])
    elif not isinstance(data, torch.Tensor):
        raise TypeError(f"Can only find the batch size of tensors but got {type(data)}.")
    return data.shape[0]


def _gpu_gather_object(object: Any):
    output_objects = [None for _ in range(RuntimeState().num_processes)]
    torch.distributed.all_gather_object(output_objects, object)
    # all_gather_object returns a list of lists, so we need to flatten it
    return [x for y in output_objects for x in y]


def gather_object(object: Any):
    if RuntimeState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        return _gpu_gather_object(object)
    else:
        return object


def _gpu_broadcast(data, src=0):
    def _gpu_broadcast_one(tensor, src=0):
        torch.distributed.broadcast(tensor, src=src)
        return tensor

    return recursively_apply(_gpu_broadcast_one, data, error_on_other_type=True, src=src)


@verify_operation
def broadcast(tensor, from_process: int = 0):
    if RuntimeState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        return _gpu_broadcast(tensor, src=from_process)
    else:
        return tensor
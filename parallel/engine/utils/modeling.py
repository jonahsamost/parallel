import torch


def id_tensor_storage(tensor: torch.Tensor) -> tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.
    """
    _SIZE = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
    }
    try:
        storage_ptr = tensor.untyped_storage().data_ptr()
        storage_size = tensor.untyped_storage().nbytes()
    except Exception:
        try:
            # Fallback for torch==1.10
            storage_ptr = tensor.storage().data_ptr()
            storage_size = tensor.storage().size() * _SIZE[tensor.dtype]
        except NotImplementedError:
            # Fallback for meta storage
            storage_ptr = 0
            # On torch >=2.0 this is the tensor size
            storage_size = tensor.nelement() * _SIZE[tensor.dtype]

    return tensor.device, storage_ptr, storage_size
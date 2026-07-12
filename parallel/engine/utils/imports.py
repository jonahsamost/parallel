import importlib


def _is_package_available(pkg_name):
    pkg = importlib.util.find_spec(pkg_name)
    if pkg is not None:
        try:
            _ = importlib.metadata.metadata(pkg_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

def is_wandb_available():
    return _is_package_available("wandb")


def is_deepspeed_available():
    return _is_package_available("deepspeed")

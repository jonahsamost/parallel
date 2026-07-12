

from parallel.engine.utils.dataclasses import DistType


def get_active_deepspeed_plugin(state):
    if state.distributed_type != DistType.DEEPSPEED:
        raise ValueError("State is not configured to use DeepSpeed")
    if not isinstance(state.deepspeed_plugins, dict):
        return state.deepspeed_plugins
    return next(plugin for plugin in state.deepspeed_plugins.values() if plugin.selected)
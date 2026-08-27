"""Online RoboTwin policy adapters."""

from importlib import import_module

__all__ = [
    "InferenceMode",
    "RoboNanaRobotWinPolicy",
    "postprocess_action",
    "preprocess_action_chunk",
]


def __getattr__(name: str):
    if name in __all__:
        return getattr(import_module(".robotwin_policy", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

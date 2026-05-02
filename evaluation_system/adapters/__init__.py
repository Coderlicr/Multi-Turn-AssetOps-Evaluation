"""Pluggable adapters: external JSON -> canonical DialogRecord."""

from evaluation_system.adapters.base import AdapterResult, BaseDialogAdapter
from evaluation_system.adapters.example import ExampleDialogAdapter
from evaluation_system.adapters.normalize import normalize_dialog
from evaluation_system.adapters.registry import AdapterRegistry, default_registry

__all__ = [
    "AdapterResult",
    "BaseDialogAdapter",
    "ExampleDialogAdapter",
    "AdapterRegistry",
    "default_registry",
    "normalize_dialog",
]

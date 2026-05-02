"""
Single entry-point for converting arbitrary external payloads.

Callers should prefer this function over instantiating adapters directly so
adapter selection stays consistent across CLI / services / notebooks.
"""

from __future__ import annotations

from typing import Any, Optional

from evaluation_system.adapters.base import AdapterResult, BaseDialogAdapter
from evaluation_system.adapters.registry import AdapterRegistry, default_registry
from evaluation_system.models.enums import ParseMode


def normalize_dialog(
    raw: Any,
    *,
    adapter_name: Optional[str] = None,
    mode: ParseMode = ParseMode.LENIENT,
    registry: Optional[AdapterRegistry] = None,
    adapter: Optional[BaseDialogAdapter] = None,
) -> AdapterResult:
    """
    Normalize external `raw` data to DialogRecord.

    Provide either `adapter` (direct instance) OR use registry resolution via
    `adapter_name` / hints inside raw / defaults.

    TODO: Support adapter_name embedded per-row when batching mixed formats.
    """
    reg = registry or default_registry()
    if adapter is not None:
        return adapter.adapt(raw, mode=mode)

    name = reg.resolve_name(raw, adapter_name)
    ad = reg.get(name)
    return ad.adapt(raw, mode=mode)

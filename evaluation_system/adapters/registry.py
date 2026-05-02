"""Registry / factory for dialog format adapters."""

from __future__ import annotations

from typing import Any, Optional

from evaluation_system.adapters.base import BaseDialogAdapter
from evaluation_system.adapters.event_stream import AssetOpsEventStreamAdapter
from evaluation_system.adapters.example import ExampleDialogAdapter
from evaluation_system.adapters.qa_history_rollout import QAHistoryRolloutAdapter


class AdapterRegistry:
    """
    Central place to register adapters.

    Typical extension: instantiate your registry (or extend default_registry)
    and `register` a new adapter for each external JSON schema version.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, BaseDialogAdapter] = {}

    def register(self, adapter: BaseDialogAdapter, *, overwrite: bool = False) -> None:
        key = adapter.name
        if key in self._by_name and not overwrite:
            raise KeyError(f"Adapter {key!r} already registered")
        self._by_name[key] = adapter

    def get(self, name: str) -> BaseDialogAdapter:
        try:
            return self._by_name[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._by_name)) or "<none>"
            raise KeyError(f"Unknown adapter {name!r}. Known: {known}") from exc

    def resolve_name(self, raw: object, explicit: Optional[str] = None) -> str:
        """
        Pick adapter name.

        Priority:
        1) explicit parameter
        2) raw["adapter_name"] if str (convention for batch files)
        3) bundled AssetOps event-stream adapter (the project's primary input format)
        """
        if explicit:
            return explicit
        if isinstance(raw, dict):
            hint = raw.get("adapter_name") or raw.get("eval_adapter")
            if isinstance(hint, str) and hint:
                return hint
        return AssetOpsEventStreamAdapter().name


def default_registry(
    *,
    dialog_specs: Optional[dict[str, dict[str, Any]]] = None,
) -> AdapterRegistry:
    """Build a registry pre-populated with the project's adapters.

    ``dialog_specs`` (the parsed ``data/dialog_specs.json``) is needed by the
    AssetOps event-stream adapter to attach ground truth from DESIGN.md.
    """
    reg = AdapterRegistry()
    reg.register(ExampleDialogAdapter(), overwrite=True)
    reg.register(AssetOpsEventStreamAdapter(dialog_specs=dialog_specs), overwrite=True)
    reg.register(QAHistoryRolloutAdapter(dialog_specs=dialog_specs), overwrite=True)
    return reg

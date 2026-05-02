"""
Example adapter for the *final canonical JSON input schema*.

This adapter is intentionally straightforward: it validates and coerces the
canonical schema into the internal `DialogRecord` Pydantic model.

Future teammate-specific formats should be implemented as *new* adapters
(keeping the main pipeline strictly depending on `DialogRecord`).
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from evaluation_system.adapters.base import AdapterResult, BaseDialogAdapter
from evaluation_system.models.dialog import DialogRecord
from evaluation_system.models.enums import ParseMode


class ExampleDialogAdapter(BaseDialogAdapter):
    """
    Canonical schema adapter.

    - `STRICT`: invalid / missing required fields raise.
    - `LENIENT`: best-effort coercion; missing required fields still raise if
      Pydantic cannot validate.
    """

    @property
    def name(self) -> str:
        return "canonical_v1"

    def adapt(self, raw: Any, *, mode: ParseMode = ParseMode.LENIENT) -> AdapterResult:
        warnings: list[str] = []
        if not isinstance(raw, dict):
            raise TypeError(f"canonical_v1 expects dict root, got {type(raw)}")
        try:
            dialog = DialogRecord.model_validate(raw)
        except ValidationError as exc:
            if mode is ParseMode.STRICT:
                raise
            # LENIENT: still raise, but wrap into a clearer error message.
            raise ValueError(f"Payload is not valid canonical_v1 dialog JSON: {exc}") from exc
        return AdapterResult(dialog=dialog, warnings=warnings)

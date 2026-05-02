"""
Abstract adapter interface.

All ingestion of external agent outputs should flow through an adapter
implementation to preserve a stable internal representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from evaluation_system.models.dialog import DialogRecord
from evaluation_system.models.enums import ParseMode


@dataclass
class AdapterResult:
    """Outcome of adapting external data to a DialogRecord."""

    dialog: DialogRecord
    warnings: list[str] = field(default_factory=list)


class BaseDialogAdapter(ABC):
    """
    Base class for format-specific adapters.

    Subclasses should:
    - Accept configuration (field mappings, optional coercion hooks)
    - Validate / coerce external payloads
    - Return AdapterResult with human-readable warnings when data is incomplete

    TODO: Add optional schema version detection (e.g., raw["format_version"]).
    TODO: Add streaming / incremental ingestion for very large logs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable registry key for this adapter."""

    @abstractmethod
    def adapt(self, raw: Any, *, mode: ParseMode = ParseMode.LENIENT) -> AdapterResult:
        """
        Convert arbitrary external representation into DialogRecord.

        Args:
            raw: Typically dict from json.load; could be str for some formats.
            mode: STRICT surfaces missing critical fields; LENIENT fills defaults.
        """

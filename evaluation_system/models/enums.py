"""Enumerations shared across the evaluation system (canonical schema aligned)."""

from __future__ import annotations

from enum import Enum


class Complexity(str, Enum):
    """Coarse scenario difficulty label."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNKNOWN = "Unknown"


class TurnStatus(str, Enum):
    """Lifecycle / outcome status for a single turn."""

    SUCCESS = "success"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ParseMode(str, Enum):
    """
    Adapter parsing behavior.

    STRICT: missing required mapped fields raise errors.
    LENIENT: fill defaults where possible and record warnings.
    """

    STRICT = "strict"
    LENIENT = "lenient"


class EvalSource(str, Enum):
    """Evaluation signal source for aggregation."""

    AUTOMATIC = "automatic"
    LLM = "llm"
    HUMAN = "human"

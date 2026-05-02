#!/usr/bin/env python3
"""Project CLI entry (run from repo root: python main.py --help)."""


def _load_dotenv_optional() -> None:
    """Load repo-root `.env` into ``os.environ`` if present (no extra dependency).

    Only sets variables that are not already exported — shell env wins.

    Typical line: OPENAI_API_KEY=sk-...
    """
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parent
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


_load_dotenv_optional()

from evaluation_system.cli import main

if __name__ == "__main__":
    main()

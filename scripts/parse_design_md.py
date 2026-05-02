#!/usr/bin/env python3
"""
Parse DESIGN.md (the 16 dialog scenario spec) into ``data/dialog_specs.json``.

Thin wrapper around :func:`evaluation_system.io.design_md.write_dialog_specs`
so the same logic can be invoked either via this CLI or from
``python main.py run-all``.

Usage::

    python scripts/parse_design_md.py --design ./data/DESIGN.md --out ./data/dialog_specs.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so ``evaluation_system`` is importable
# when this script is invoked directly (e.g. ``python scripts/parse_design_md.py``).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_system.io.design_md import write_dialog_specs  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", required=True, type=Path, help="Path to DESIGN.md")
    p.add_argument("--out", required=True, type=Path, help="Output dialog_specs.json")
    p.add_argument(
        "--annotated",
        type=Path,
        default=None,
        help="Optional DESIGN_annotated.md (default: <design parent>/DESIGN_annotated.md if present).",
    )
    p.add_argument(
        "--no-annotated-overlay",
        action="store_true",
        help="Do not merge DESIGN_annotated even if present next to DESIGN.md.",
    )
    args = p.parse_args()

    try:
        ann: Path | None
        if args.no_annotated_overlay:
            ann = None
        elif args.annotated is not None:
            ann = args.annotated if args.annotated.is_file() else None
            if args.annotated and ann is None:
                print(f"warning: --annotated not found: {args.annotated}; skipping overlay.", file=sys.stderr)
        else:
            cand = args.design.parent / "DESIGN_annotated.md"
            ann = cand if cand.is_file() else None
        n, merged = write_dialog_specs(args.design, args.out, annotated_md=ann)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    msg = f"Parsed {n} dialog spec(s) -> {args.out}"
    if merged:
        msg += f" ({merged} with DESIGN_annotated overlay)"
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

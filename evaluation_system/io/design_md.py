"""
Parse DESIGN.md (the 16 dialog scenario spec) into structured ground-truth specs.

The output dict (keyed by integer dialog_id as string) has the schema below
and is what ``data/dialog_specs.json`` carries::

    {
      "1": {
        "dialog_id": 1,
        "scenario_type": "Fault Diagnosis + Maintenance",
        "scenario_subtype": "Temperature Anomaly Diagnosis ...",
        "complexity": "High",
        "tool_domains": ["IoT", "TSFM", "FMSR"],
        "key_capabilities": ["Anomaly detection", ...],
        "turns": [
          {"turn_id": 1, "speaker": "User",   "text": "...", "tool_domains": []},
          {"turn_id": 1, "speaker": "System", "text": "...", "tool_domains": ["IoT"]},
          ...
        ],
        "ground_truth": {
          "expected_tools":          [...],
          "expected_plan":           [...],
          "expected_final_answer":   "...",
          "task_success_criteria":   [...],
          "acceptable_alternatives": [],
          "annotated_characteristic": "...",
          "required_tool_sequence":   "Markdown table text from DESIGN_annotated.md"
        }

Optional keys ``annotated_characteristic`` / ``required_tool_sequence``, and finer-grained
``task_success_criteria`` bullets, are merged when an annotated design file (e.g.
``DESIGN_annotated.md``) is provided to :func:`write_dialog_specs`.

      },
      ...
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


_DIALOG_HEADING_RE = re.compile(r"^##\s+\*\*Dialog\s+(\d+):\s+(.+?)\*\*\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\|\s*:?-{3,}.*\|\s*$")


def _split_md_cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    raw = s.split("|")
    return [c.strip().replace(r"\|", "|").replace(r"\#", "#").replace(r"\+", "+") for c in raw]


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def _scan_dialog_blocks(md: str) -> list[tuple[int, str, list[str]]]:
    blocks: list[tuple[int, str, list[str]]] = []
    cur_id: Optional[int] = None
    cur_title = ""
    cur_lines: list[str] = []
    for line in md.splitlines():
        m = _DIALOG_HEADING_RE.match(line)
        if m:
            if cur_id is not None:
                blocks.append((cur_id, cur_title, cur_lines))
            cur_id = int(m.group(1))
            cur_title = m.group(2).strip()
            cur_lines = []
        else:
            if cur_id is not None:
                cur_lines.append(line)
    if cur_id is not None:
        blocks.append((cur_id, cur_title, cur_lines))
    return blocks


def _extract_metadata(body_lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in body_lines:
        if not _is_table_row(line) or _TABLE_SEPARATOR_RE.match(line):
            continue
        cells = _split_md_cells(line)
        if len(cells) != 2:
            break
        k = cells[0].strip("* ").strip()
        v = cells[1].strip("* ").strip()
        if not k or k.lower() == "category" and out:
            break
        out[k.lower()] = v
        if k.lower() == "related scenarios":
            break
    return out


def _extract_turns(body_lines: list[str]) -> list[dict[str, object]]:
    turns: list[dict[str, object]] = []
    in_table = False
    saw_separator = False
    for line in body_lines:
        if not _is_table_row(line):
            if in_table and turns:
                break
            in_table = False
            saw_separator = False
            continue
        cells = _split_md_cells(line)
        if (
            len(cells) == 4
            and cells[0].lower().strip(" *") == "turn"
            and "header" in cells[1].lower()
        ):
            in_table = True
            saw_separator = False
            continue
        if not in_table:
            continue
        if _TABLE_SEPARATOR_RE.match(line):
            saw_separator = True
            continue
        if not saw_separator or len(cells) != 4:
            continue
        turn_id_raw = cells[0].strip(" *")
        try:
            turn_id = int(turn_id_raw)
        except ValueError:
            continue
        speaker = cells[1].strip(" *")
        text = cells[2].strip()
        domains_raw = cells[3].strip()
        domains = [d.strip() for d in domains_raw.split(",") if d.strip()] if domains_raw else []
        turns.append(
            {
                "turn_id": turn_id,
                "speaker": speaker,
                "text": text,
                "tool_domains": domains,
            }
        )
    return turns


def _split_capabilities(s: str) -> list[str]:
    parts = [p.strip(" .") for p in re.split(r",|;", s) if p.strip()]
    return [p for p in parts if p]


def _split_tools(s: str) -> list[str]:
    parts = [p.strip(" .") for p in re.split(r",|/|\+|&", s) if p.strip()]
    return [p for p in parts if p]


def _build_ground_truth(
    *,
    metadata: dict[str, str],
    turns: list[dict[str, object]],
) -> dict[str, object]:
    expected_tools = _split_tools(metadata.get("tool domains involved", ""))
    expected_plan = [str(t["text"]) for t in turns if t["speaker"].lower() == "system"]
    expected_final_answer = expected_plan[-1] if expected_plan else ""
    success_criteria = _split_capabilities(metadata.get("key capabilities", ""))
    return {
        "expected_tools": expected_tools,
        "expected_plan": expected_plan,
        "expected_final_answer": expected_final_answer,
        "task_success_criteria": success_criteria,
        "acceptable_alternatives": [],
    }


def _squish_blank_lines(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def _parse_success_criteria_bullets(blob: str) -> list[str]:
    out: list[str] = []
    for line in blob.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^[-*]\s*", s):
            cleaned = re.sub(r"^[-*]\s*(\[[ x]\]\s*)?", "", s).strip()
            if cleaned:
                out.append(cleaned)
    return out


def _extract_annotated_sections_from_dialog_body(body: str) -> Optional[dict[str, object]]:
    """
    Pull ``### **Ground Truth**`` subsections: Characteristic Form, Required Tool Sequence,
    Success Criteria. Returns None if there is no Ground Truth block.
    """
    if not re.search(r"###\s+\*\*Ground\s+Truth\*\*", body, flags=re.IGNORECASE):
        return None
    m_gt = re.search(r"###\s+\*\*Ground\s+Truth\*\*", body, flags=re.IGNORECASE)
    assert m_gt is not None
    after = body[m_gt.end() :]

    m_char = re.search(
        r"\*\*Characteristic\s+Form\*\*\s*(.*?)(?=\*\*Required\s+Tool\s+Sequence\*\*)",
        after,
        flags=re.DOTALL | re.IGNORECASE,
    )
    m_req = re.search(
        r"\*\*Required\s+Tool\s+Sequence\*\*\s*(.*?)(?=\*\*Success\s+Criteria\*\*)",
        after,
        flags=re.DOTALL | re.IGNORECASE,
    )
    m_suc = re.search(r"\*\*Success\s+Criteria\*\*\s*(.*)\Z", after, flags=re.DOTALL | re.IGNORECASE)

    out: dict[str, object] = {}
    if m_char:
        out["annotated_characteristic"] = _squish_blank_lines(m_char.group(1))
    if m_req:
        out["required_tool_sequence"] = _squish_blank_lines(m_req.group(1))
    if m_suc:
        bullets = _parse_success_criteria_bullets(m_suc.group(1))
        if bullets:
            out["task_success_criteria"] = bullets
    if not out:
        return None
    return out


def parse_overlay_ground_truth_annotated(md_text: str) -> dict[int, dict[str, object]]:
    """Per-dialog overlay dicts keyed by dialog_id (from ``DESIGN_annotated.md``)."""
    overlays: dict[int, dict[str, object]] = {}
    for dialog_id, _title, body_lines in _scan_dialog_blocks(md_text):
        body = "\n".join(body_lines)
        sec = _extract_annotated_sections_from_dialog_body(body)
        if sec:
            overlays[dialog_id] = sec
    return overlays


def merge_annotated_ground_truth_overlays(
    specs: dict[str, dict[str, object]],
    overlays: dict[int, dict[str, object]],
) -> int:
    """
    In-place merge of annotated fields into each ``specs[str(id)]["ground_truth"]``.

    Returns the number of dialogs updated.
    """
    n = 0
    for did, extra in overlays.items():
        key = str(did)
        if key not in specs:
            continue
        base = specs[key]
        gt = base.get("ground_truth")
        if not isinstance(gt, dict):
            gt = {}
            base["ground_truth"] = gt
        if "annotated_characteristic" in extra:
            gt["annotated_characteristic"] = extra["annotated_characteristic"]
        if "required_tool_sequence" in extra:
            gt["required_tool_sequence"] = extra["required_tool_sequence"]
        if "task_success_criteria" in extra:
            gt["task_success_criteria"] = extra["task_success_criteria"]
        n += 1
    return n


def parse_design_md(md_text: str) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for dialog_id, subtype_title, body in _scan_dialog_blocks(md_text):
        metadata = _extract_metadata(body)
        turns = _extract_turns(body)
        scenario_type = metadata.get("category", "Unknown")
        complexity_raw = metadata.get("complexity", "Unknown")
        complexity = complexity_raw.split(" ", 1)[0].rstrip("(,").strip()
        out[str(dialog_id)] = {
            "dialog_id": dialog_id,
            "scenario_type": scenario_type,
            "scenario_subtype": subtype_title,
            "complexity": complexity,
            "tool_domains": _split_tools(metadata.get("tool domains involved", "")),
            "key_capabilities": _split_capabilities(metadata.get("key capabilities", "")),
            "turns": turns,
            "ground_truth": _build_ground_truth(metadata=metadata, turns=turns),
        }
    return out


def write_dialog_specs(
    design_md: Path,
    specs_out: Path,
    *,
    annotated_md: Optional[Path] = None,
) -> tuple[int, int]:
    """Parse ``DESIGN.md`` and write JSON to ``specs_out``.

    When ``annotated_md`` is an existing file (e.g. ``DESIGN_annotated.md``), merge
    per-dialog **Ground Truth** blocks (characteristic form, required tool sequence,
    success-criteria bullets into ``task_success_criteria``).

    Returns ``(n_dialogs, n_with_annotated_overlay)``. Raises ``FileNotFoundError``
    if ``design_md`` is missing; ``ValueError`` if no dialogs parsed.
    """
    if not design_md.is_file():
        raise FileNotFoundError(f"DESIGN.md not found: {design_md}")
    md_text = design_md.read_text(encoding="utf-8")
    parsed = parse_design_md(md_text)
    if not parsed:
        raise ValueError(f"DESIGN.md at {design_md} produced no dialog specs.")
    merged_n = 0
    if annotated_md is not None and annotated_md.is_file():
        amo = annotated_md.read_text(encoding="utf-8")
        overlays = parse_overlay_ground_truth_annotated(amo)
        merged_n = merge_annotated_ground_truth_overlays(parsed, overlays)
    specs_out.parent.mkdir(parents=True, exist_ok=True)
    specs_out.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(parsed), merged_n

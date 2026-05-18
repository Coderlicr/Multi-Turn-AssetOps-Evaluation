"""
CLI for the multi-turn dialog evaluation system.

Subcommands:
1) normalize           Convert one external JSON/JSONL to canonical DialogRecord
2) auto-evaluate       Compute objective metrics over a results tree
3) llm-judge           Run LLM-as-Judge over dialogs, writing llm_evaluation
4) export-annotation   Emit CSV + Markdown templates for human review
5) import-annotation   Apply human annotation CSV back into dialog records
6) leaderboard         Aggregate per-model metrics into the final table
7) visualize           Render radar comparison figure from judged dialogs
8) prepare-judge-training  Build OpenAI fine-tuning JSONL from labeled dialogs
9) run-all             One-shot: DESIGN.md → llm-judge → leaderboard → visualize
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from evaluation_system.adapters.normalize import normalize_dialog
from evaluation_system.adapters.registry import default_registry
from evaluation_system.evaluators.human_annotation.annotation import (
    apply_human_annotation,
    export_annotation_csv_row,
    export_annotation_markdown,
    read_annotations_csv,
)
from evaluation_system.evaluators.llm_judge.config import JudgeRunConfig
from evaluation_system.evaluators.llm_judge.judge import (
    BaseJudge,
    ChatCompletionError,
    MockJudge,
    OpenAIJudge,
)
from evaluation_system.evaluators.llm_judge.parser import JudgeOutputParseError
from evaluation_system.evaluators.llm_judge.runner import run_llm_judge
from evaluation_system.evaluators.llm_judge.training import (
    build_judge_training_examples,
    split_train_val,
    write_openai_finetune_jsonl,
)
from evaluation_system.io.file_ops import (
    DEFAULT_DIALOG_GLOB,
    discover_model_result_directories,
    export_dialog_json,
    load_dialog_specs,
    load_one_dialog_autodetect,
    write_text,
)
from evaluation_system.leaderboard.table import (
    build_aggregate_leaderboard,
    export_leaderboard_csv,
    export_leaderboard_json,
    format_leaderboard_text,
)
from evaluation_system.metrics.aggregation import aggregate_metrics
from evaluation_system.models.dialog import DialogRecord
from evaluation_system.models.enums import EvalSource, ParseMode


DEFAULT_DATA_ROOT = Path("./data")
DEFAULT_DIALOG_SPECS_PATH = DEFAULT_DATA_ROOT / "dialog_specs.json"
DEFAULT_DESIGN_MD_PATH = DEFAULT_DATA_ROOT / "DESIGN.md"

# OpenAI-backed judge defaults (override via --model).
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-4o-mini"


def _parse_mode(strict: bool) -> ParseMode:
    return ParseMode.STRICT if strict else ParseMode.LENIENT


def _iter_dialog_files(model_dir: Path, pattern: str) -> list[Path]:
    return [p for p in sorted(model_dir.glob(pattern)) if p.is_file()]


def _build_registry(ns: argparse.Namespace):
    """Build a registry pre-loaded with dialog ground-truth specs.

    The dialog_specs path is configurable via ``--specs`` and falls back to
    ``./data/dialog_specs.json`` (silent if that file is absent — adapters
    will warn per-file when a dialog_id has no matching spec).
    """
    specs_arg: Optional[str] = getattr(ns, "specs", None)
    specs_path = Path(specs_arg) if specs_arg else DEFAULT_DIALOG_SPECS_PATH
    specs = load_dialog_specs(specs_path)
    return default_registry(dialog_specs=specs)


def _load_one(ns: argparse.Namespace, fp: Path, *, registry, model_name: str) -> DialogRecord:
    return load_one_dialog_autodetect(
        fp,
        model_name=model_name,
        adapter_name=ns.adapter,
        mode=_parse_mode(ns.strict),
        registry=registry,
    )


def cmd_normalize(ns: argparse.Namespace) -> None:
    inp = Path(ns.input)
    if not inp.is_file():
        raise FileNotFoundError(f"Not a file: {inp}")
    registry = _build_registry(ns)
    model_name = inp.parent.name
    d = load_one_dialog_autodetect(
        inp,
        model_name=model_name,
        adapter_name=ns.adapter,
        mode=_parse_mode(ns.strict),
        registry=registry,
    )
    out_json = json.dumps(d.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if ns.out:
        Path(ns.out).write_text(out_json + "\n", encoding="utf-8")
    else:
        print(out_json)


def cmd_auto_evaluate(ns: argparse.Namespace) -> None:
    root = Path(ns.input)
    out_root = Path(ns.out) if ns.out else root / "_evaluated"
    models = discover_model_result_directories(root)
    registry = _build_registry(ns)
    total = 0
    for model_name, dir_path in models:
        model_out = out_root / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        count = 0
        for fp in _iter_dialog_files(dir_path, ns.pattern):
            d = _load_one(ns, fp, registry=registry, model_name=model_name)
            export_dialog_json(d, model_out / f"{d.dialog_id}.json")
            count += 1
        total += count
        print(f"  model={model_name!r}: {count} dialog(s) -> {model_out}")
    print(f"Wrote auto-evaluated outputs for {len(models)} model(s), {total} dialog(s) under {out_root}")


def _canonical_dest_name(src: Path) -> str:
    """Persisted DialogRecord output is always canonical ``.json``.

    Raw rollout ``.jsonl`` files become ``<stem>.json`` after judging so the
    next pipeline stage doesn't try to re-parse them as event streams.
    """
    return src.with_suffix(".json").name


def _llm_judge_targets(
    inp: Path, pattern: str, out_root: Path | None
) -> list[tuple[Path, Path, str | None]]:
    """
    (source_path, dest_path, model_folder_name or None).

    - Single file: one tuple; dest is ``out_root`` when set, else next to
      the source as ``<stem>.json``.
    - Directory: uses ``discover_model_result_directories`` layout, and
      always writes output as ``<stem>.json``.
    """
    if inp.is_file():
        if out_root is None:
            dest = inp.with_name(_canonical_dest_name(inp))
        elif out_root.is_dir():
            dest = out_root / _canonical_dest_name(inp)
        else:
            dest = out_root
        return [(inp, dest, None)]

    models = discover_model_result_directories(inp)
    out: list[tuple[Path, Path, str | None]] = []
    for model_name, dir_path in models:
        for fp in _iter_dialog_files(dir_path, pattern):
            dest_name = _canonical_dest_name(fp)
            if out_root is not None:
                dest = out_root / model_name / dest_name
            else:
                dest = fp.with_name(dest_name)
            out.append((fp, dest, model_name))
    return out


def _paired_llm_judge_targets(
    inp: Path, pattern: str, out_root: Path | None
) -> list[tuple[str, list[tuple[str, Path, Path]]]]:
    """
    Group directory inputs by dialog filename stem across model folders.

    Returns ``[(dialog_key, [(model_name, source_path, dest_path), ...]), ...]``.
    """
    models = discover_model_result_directories(inp)
    by_key: dict[str, list[tuple[str, Path, Path]]] = {}
    for model_name, dir_path in models:
        for fp in _iter_dialog_files(dir_path, pattern):
            dest_name = _canonical_dest_name(fp)
            dest = out_root / model_name / dest_name if out_root is not None else fp.with_name(dest_name)
            by_key.setdefault(fp.stem, []).append((model_name, fp, dest))

    out: list[tuple[str, list[tuple[str, Path, Path]]]] = []
    for key in sorted(by_key):
        out.append((key, sorted(by_key[key], key=lambda item: item[0])))
    return out


def _candidate_label(idx: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if idx < len(alphabet):
        return f"candidate_{alphabet[idx]}"
    return f"candidate_{idx + 1}"


def _build_judge(ns: argparse.Namespace, run_cfg: JudgeRunConfig) -> BaseJudge:
    if ns.judge == "mock":
        return MockJudge(run_config=run_cfg)
    if ns.judge == "openai":
        model_id = (getattr(ns, "model", None) or "").strip() or DEFAULT_OPENAI_JUDGE_MODEL
        return OpenAIJudge(
            model_id,
            temperature=float(ns.temperature),
            timeout_sec=float(ns.timeout),
            max_retries=int(ns.retries),
            base_url=ns.base_url or None,
            organization=ns.organization or None,
            run_config=run_cfg,
        )
    raise ValueError(f"Unknown judge: {ns.judge!r}")


def cmd_llm_judge(ns: argparse.Namespace) -> None:
    inp = Path(ns.input)
    out_arg = Path(ns.out) if ns.out else None

    run_cfg = JudgeRunConfig(
        runs=int(ns.runs),
        shuffle_rubric=not ns.no_shuffle,
        seed=int(ns.seed),
        prompt_version=str(ns.prompt_version),
        blind_model_name=bool(ns.blind_model_name),
    )
    judge = _build_judge(ns, run_cfg)
    registry = _build_registry(ns)
    judge_mode = str(getattr(ns, "judge_mode", "paired"))

    if judge_mode == "paired" and inp.is_dir():
        groups = _paired_llm_judge_targets(inp, ns.pattern, out_arg)
        if not groups:
            raise ValueError(f"No dialog files found under {inp}")

        total = len(groups)
        failures = 0
        for i, (dialog_key, items) in enumerate(groups, start=1):
            prefix = f"[{i}/{total}]"
            try:
                loaded: list[tuple[str, str, Path, DialogRecord]] = []
                for j, (model_name, src, dest) in enumerate(items):
                    d = _load_one(ns, src, registry=registry, model_name=model_name)
                    loaded.append((_candidate_label(j), model_name, dest, d))

                if len(loaded) >= 2:
                    evaluations = judge.judge_pair([(label, d) for label, _, _, d in loaded])
                    mode_note = "paired"
                else:
                    evaluations = {loaded[0][0]: judge.judge(loaded[0][3])}
                    mode_note = "single"

                for label, model_name, dest, d in loaded:
                    d2 = d.model_copy(update={"llm_evaluation": evaluations[label]})
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    export_dialog_json(d2, dest)

                locs = ", ".join(f"{model_name}/{dest.name}" for _, model_name, dest, _ in loaded)
                print(f"{prefix} ok {dialog_key} ({mode_note}) -> {locs}")
            except (ChatCompletionError, JudgeOutputParseError, OSError, ValueError, ValidationError) as exc:
                failures += 1
                print(f"{prefix} error {dialog_key}: {exc}", file=sys.stderr)
                if not getattr(ns, "continue_on_error", False):
                    raise SystemExit(2) from exc
        if failures:
            raise SystemExit(1)
        return

    targets = _llm_judge_targets(inp, ns.pattern, out_arg)
    if not targets:
        raise ValueError(f"No dialog files found under {inp}")

    total = len(targets)
    failures = 0
    for i, (src, dest, model_name) in enumerate(targets, start=1):
        prefix = f"[{i}/{total}]"
        try:
            d = _load_one(
                ns,
                src,
                registry=registry,
                model_name=model_name or src.parent.name,
            )
            d2 = run_llm_judge(d, judge=judge)
            dest.parent.mkdir(parents=True, exist_ok=True)
            export_dialog_json(d2, dest)
            loc = f"{model_name}/" if model_name else ""
            print(f"{prefix} ok {loc}{src.name} -> {dest}")
        except (ChatCompletionError, JudgeOutputParseError, OSError, ValueError, ValidationError) as exc:
            failures += 1
            print(f"{prefix} error {src}: {exc}", file=sys.stderr)
            if not getattr(ns, "continue_on_error", False):
                raise SystemExit(2) from exc
    if failures:
        raise SystemExit(1)


def cmd_export_annotation(ns: argparse.Namespace) -> None:
    root = Path(ns.input)
    out_dir = Path(ns.out) if ns.out else root / "_annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = _build_registry(ns)
    models = discover_model_result_directories(root)
    csv_path = out_dir / "annotations_template.csv"
    md_dir = out_dir / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_name, dir_path in models:
        for fp in _iter_dialog_files(dir_path, ns.pattern):
            d = _load_one(ns, fp, registry=registry, model_name=model_name)
            rows.append(export_annotation_csv_row(d))
            write_text(md_dir / f"{d.dialog_id}.md", export_annotation_markdown(d))

    if not rows:
        raise ValueError(f"No dialogs found under {root}")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote annotation templates: {csv_path}")
    print(f"Wrote markdown previews: {md_dir}")


def cmd_import_annotation(ns: argparse.Namespace) -> None:
    ann_path = Path(ns.input)
    results_root = Path(ns.results)
    out_root = Path(ns.out) if ns.out else results_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows = read_annotations_csv(ann_path)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        by_key[(r.model_name, r.dialog_id)] = {
            "annotator_id": r.annotator_id,
            "planning_effectiveness": r.planning_effectiveness,
            "tool_usage_quality": r.tool_usage_quality,
            "task_completion": r.task_completion,
            "comments": r.comments,
        }

    registry = _build_registry(ns)
    models = discover_model_result_directories(results_root)
    updated = 0
    for model_name, dir_path in models:
        model_out = (out_root / model_name) if out_root != results_root else dir_path
        model_out.mkdir(parents=True, exist_ok=True)
        for fp in _iter_dialog_files(dir_path, ns.pattern):
            d = _load_one(ns, fp, registry=registry, model_name=model_name)
            key = (d.model_name or model_name, d.dialog_id)
            ann = by_key.get(key)
            if ann is None:
                export_dialog_json(d, model_out / fp.name)
                continue
            d2 = apply_human_annotation(d, ann)
            export_dialog_json(d2, model_out / fp.name)
            updated += 1
    print(f"Applied human annotations to {updated} dialog(s). Output: {out_root}")


def cmd_leaderboard(ns: argparse.Namespace) -> None:
    root = Path(ns.input)
    out_dir = Path(ns.out) if ns.out else root / "_leaderboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = _build_registry(ns)
    models = discover_model_result_directories(root)
    source = EvalSource(ns.metric_source)
    per_model = {}
    for model_name, dir_path in models:
        ds: list[DialogRecord] = []
        for fp in _iter_dialog_files(dir_path, ns.pattern):
            ds.append(_load_one(ns, fp, registry=registry, model_name=model_name))
        if not ds:
            print(f"  model={model_name!r}: no dialog files matched {ns.pattern!r}; skipping",
                  file=sys.stderr)
            continue
        per_model[model_name] = aggregate_metrics(ds, source=source)
    if not per_model:
        raise ValueError(f"No model directory under {root} produced any dialogs.")
    rows = build_aggregate_leaderboard(per_model)
    table = format_leaderboard_text(rows)
    print(table)
    export_leaderboard_json(rows, out_dir / "leaderboard_metrics.json")
    export_leaderboard_csv(rows, out_dir / "leaderboard_metrics.csv")
    write_text(out_dir / "leaderboard_metrics.txt", table + "\n")
    print(f"\nLeaderboard files written under {out_dir}")


def _aggregate_per_model(
    ns: argparse.Namespace,
    *,
    source: EvalSource,
):
    """Read every dialog under ``ns.input`` and return per-model AggregatedMetrics."""
    root = Path(ns.input)
    registry = _build_registry(ns)
    models = discover_model_result_directories(root)
    per_model = {}
    for model_name, dir_path in models:
        ds: list[DialogRecord] = []
        for fp in _iter_dialog_files(dir_path, ns.pattern):
            ds.append(_load_one(ns, fp, registry=registry, model_name=model_name))
        if not ds:
            print(f"  model={model_name!r}: no dialog files matched {ns.pattern!r}; skipping",
                  file=sys.stderr)
            continue
        per_model[model_name] = aggregate_metrics(ds, source=source)
    if not per_model:
        raise ValueError(f"No model directory under {root} produced any dialogs.")
    return per_model


def cmd_visualize(ns: argparse.Namespace) -> None:
    """Render comparison figures (PNG) from judged dialog records."""
    try:
        from evaluation_system.leaderboard.figures import plot_all
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for `visualize`. Install with: pip install matplotlib"
        ) from exc

    out_dir = Path(ns.out) if ns.out else Path(ns.input) / "_figures"
    per_model = _aggregate_per_model(ns, source=EvalSource(ns.metric_source))
    paths = plot_all(per_model, out_dir)
    print(f"Wrote {len(paths)} figure(s) under {out_dir}:")
    for p in paths:
        print(f"  - {p}")


def cmd_prepare_judge_training(ns: argparse.Namespace) -> None:
    """Convert annotated/judged dialog records into OpenAI fine-tuning JSONL."""
    root = Path(ns.input)
    out_path = Path(ns.out)
    val_path = Path(ns.val_out) if ns.val_out else None
    val_fraction = float(ns.val_fraction)
    if val_fraction > 0 and val_path is None:
        val_path = out_path.with_name(f"{out_path.stem}.val{out_path.suffix or '.jsonl'}")

    registry = _build_registry(ns)
    models = discover_model_result_directories(root)
    dialogs: list[DialogRecord] = []
    for model_name, dir_path in models:
        for fp in _iter_dialog_files(dir_path, ns.pattern):
            dialogs.append(_load_one(ns, fp, registry=registry, model_name=model_name))
    if not dialogs:
        raise ValueError(f"No dialog files discovered under {root}")

    examples = build_judge_training_examples(
        dialogs,
        source=ns.source,
        prompt_version=str(ns.prompt_version),
    )
    if not examples:
        raise ValueError(
            f"No usable examples found from source={ns.source!r}. "
            "For source=human, fill all three scores via import-annotation first; "
            "for source=llm, run llm-judge with a teacher model first."
        )

    if val_fraction > 0:
        train, val = split_train_val(examples, val_fraction=val_fraction, seed=int(ns.seed))
    else:
        train, val = examples, []

    n_train = write_openai_finetune_jsonl(train, out_path)
    print(f"Wrote {n_train} training example(s) -> {out_path}")
    if val_path and val:
        n_val = write_openai_finetune_jsonl(val, val_path)
        print(f"Wrote {n_val} validation example(s) -> {val_path}")
    skipped = len(dialogs) - len(examples)
    if skipped:
        print(f"Skipped {skipped} dialog(s) without complete {ns.source} scores.")


def _resolve_annotated_design_path(design_md: Path, ns: argparse.Namespace) -> Optional[Path]:
    """Return path to DESIGN_annotated overlay, or ``None`` to skip merge."""
    if getattr(ns, "no_annotated_overlay", False):
        return None
    explicit = getattr(ns, "annotated_design", None)
    if explicit:
        p = Path(str(explicit))
        if p.is_file():
            return p
        print(f"warning: --annotated-design not found: {p}; skipping annotated overlay.", file=sys.stderr)
        return None
    cand = design_md.parent / "DESIGN_annotated.md"
    return cand if cand.is_file() else None


def _ensure_dialog_specs(design: Path, specs: Path, *, annotated_design: Optional[Path] = None) -> None:
    """Refresh ``specs`` from ``DESIGN.md`` and optionally merge ``DESIGN_annotated`` ground truth."""
    from evaluation_system.io.design_md import write_dialog_specs

    n, merged = write_dialog_specs(design, specs, annotated_md=annotated_design)
    print(f"  parsed DESIGN.md -> {specs} ({n} dialog spec(s))")
    if merged:
        print(f"  merged DESIGN_annotated ground truth for {merged} dialog(s)")


def _wipe_dir(path: Path) -> None:
    """Remove ``path`` if it exists (file or directory). Safe for fresh runs."""
    if not path.exists():
        return
    import shutil
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def cmd_run_all(ns: argparse.Namespace) -> None:
    """One-shot: DESIGN.md → llm-judge → leaderboard → visualize."""
    data_root = Path(ns.input)
    design_path = Path(ns.design) if ns.design else data_root / "DESIGN.md"
    specs_path = Path(ns.specs) if ns.specs else data_root / "dialog_specs.json"
    judged_root = Path(ns.judged) if ns.judged else data_root / "_judged"
    leaderboard_root = Path(ns.leaderboard) if ns.leaderboard else data_root / "_leaderboard"
    figures_root = Path(ns.figures) if ns.figures else leaderboard_root / "figures"
    skip_figures = bool(getattr(ns, "no_figures", False))

    # Wipe derived outputs so stale files from previous runs (especially after
    # renaming source rollouts) can never leak into aggregated metrics.
    _wipe_dir(judged_root)
    _wipe_dir(leaderboard_root)

    total = 3 if skip_figures else 4
    print(f"[1/{total}] parse DESIGN.md ({design_path})")
    _ensure_dialog_specs(design_path, specs_path, annotated_design=_resolve_annotated_design_path(design_path, ns))

    eff_model = (getattr(ns, "model", None) or "").strip() or DEFAULT_OPENAI_JUDGE_MODEL
    print(
        f"[2/{total}] llm-judge (mode=paired, judge={ns.judge}"
        + (f", model={eff_model}" if ns.judge == "openai" else "")
        + f", blind={ns.blind_model_name}"
        + f") -> {judged_root}"
    )
    judge_ns = argparse.Namespace(
        input=str(data_root),
        out=str(judged_root),
        adapter=None,
        specs=str(specs_path),
        strict=False,
        pattern=DEFAULT_DIALOG_GLOB,
        judge=ns.judge,
        judge_mode="paired",
        model=eff_model if ns.judge == "openai" else ns.model,
        runs=int(ns.runs),
        no_shuffle=False,
        seed=1337,
        prompt_version="v1.0.0",
        temperature=0.0,
        timeout=120.0,
        retries=3,
        base_url=None,
        organization=None,
        continue_on_error=False,
        blind_model_name=bool(ns.blind_model_name),
    )
    cmd_llm_judge(judge_ns)

    print(f"[3/{total}] leaderboard -> {leaderboard_root}")
    lb_ns = argparse.Namespace(
        input=str(judged_root),
        out=str(leaderboard_root),
        adapter=None,
        specs=str(specs_path),
        strict=False,
        pattern=DEFAULT_DIALOG_GLOB,
        metric_source="llm",
    )
    cmd_leaderboard(lb_ns)

    if skip_figures:
        return

    print(f"[4/{total}] visualize -> {figures_root}")
    viz_ns = argparse.Namespace(
        input=str(judged_root),
        out=str(figures_root),
        adapter=None,
        specs=str(specs_path),
        strict=False,
        pattern=DEFAULT_DIALOG_GLOB,
        metric_source="llm",
    )
    cmd_visualize(viz_ns)


def _add_common_args(p: argparse.ArgumentParser, *, with_pattern: bool = True) -> None:
    p.add_argument("--adapter", type=str, default=None,
                   help="Force a specific adapter name (default: auto-detect from input).")
    p.add_argument("--specs", type=str, default=None,
                   help="Path to dialog_specs.json (default: ./data/dialog_specs.json).")
    p.add_argument("--strict", action="store_true",
                   help="Adapter STRICT mode: missing required fields raise instead of warning.")
    if with_pattern:
        p.add_argument("--pattern", type=str, default=DEFAULT_DIALOG_GLOB,
                       help=f"Glob pattern for dialog files (default {DEFAULT_DIALOG_GLOB!r}).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluation-system",
        description=(
            "AssetOps multi-turn dialog evaluation system. "
            "Default input format: agent rollout JSONL (one event per line) "
            "joined with DESIGN.md ground truth via data/dialog_specs.json."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_norm = sub.add_parser("normalize", help="Read and normalize a single dialog file to canonical DialogRecord")
    p_norm.add_argument("--input", required=True, type=str)
    p_norm.add_argument("--out", type=str, default=None)
    _add_common_args(p_norm, with_pattern=False)
    p_norm.set_defaults(func=cmd_normalize)

    p_auto = sub.add_parser("auto-evaluate", help="Automatic evaluation over a results directory")
    p_auto.add_argument("--input", required=True, type=str,
                        help="Results root with per-model subfolders (e.g. ./data).")
    p_auto.add_argument("--out", type=str, default=None,
                        help="Output root (default: <input>/_evaluated/).")
    _add_common_args(p_auto)
    p_auto.set_defaults(func=cmd_auto_evaluate)

    p_j = sub.add_parser(
        "llm-judge",
        help="Run LLM-as-Judge on dialog records (single file or results directory).",
    )
    p_j.add_argument("--input", required=True, type=str,
                     help="Path to one dialog file or a results directory.")
    p_j.add_argument("--out", type=str, default=None,
                     help="Output path (single file mode) or output root (directory mode). Default: overwrite sources.")
    _add_common_args(p_j)
    p_j.add_argument(
        "--judge",
        type=str,
        default="openai",
        choices=["mock", "openai"],
        help=(
            "openai: real Chat Completions judge (needs OPENAI_API_KEY; optional OPENAI_BASE_URL). "
            "mock: deterministic offline heuristic (no API)."
        ),
    )
    p_j.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OPENAI_JUDGE_MODEL,
        help=(
            "OpenAI-compatible model id (default %(default)s). Ignored when --judge mock. "
            "Examples: gpt-4o-mini, gpt-4.1, o4-mini, fine-tuned ft:... ids."
        ),
    )
    p_j.add_argument("--runs", type=int, default=1, help="Multi-run averaging (default 1)")
    p_j.add_argument(
        "--judge-mode",
        type=str,
        default="paired",
        choices=["paired", "single"],
        dest="judge_mode",
        help="paired: score same-stem dialogs from different model folders in one prompt; single: score each file independently.",
    )
    p_j.add_argument("--no-shuffle", action="store_true", help="Disable rubric-order shuffling when runs>1")
    p_j.add_argument("--seed", type=int, default=1337, help="RNG seed for shuffling")
    p_j.add_argument("--prompt-version", type=str, default="v1.0.0", dest="prompt_version")
    p_j.add_argument("--temperature", type=float, default=0.0)
    p_j.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds per request")
    p_j.add_argument("--retries", type=int, default=3, help="Max retries for transient API failures")
    p_j.add_argument("--base-url", type=str, default=None, dest="base_url",
                     help="Override OpenAI base URL (Azure proxy, vLLM, OpenAI-compatible gateway).")
    p_j.add_argument("--organization", type=str, default=None,
                     help="OpenAI organization id (falls back to OPENAI_ORG / OPENAI_ORGANIZATION env vars).")
    p_j.add_argument("--continue-on-error", action="store_true", dest="continue_on_error",
                     help="Log per-file errors and continue (exit code 1 if any failed)")
    p_j.add_argument(
        "--blind-model-name",
        dest="blind_model_name",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When set (default), replace folder/model id in judge prompts with a neutral placeholder. "
            "Use --no-blind-model-name to expose real folder names."
        ),
    )
    p_j.set_defaults(func=cmd_llm_judge)

    p_exp = sub.add_parser("export-annotation", help="Export human annotation templates (CSV + Markdown)")
    p_exp.add_argument("--input", required=True, type=str)
    p_exp.add_argument("--out", type=str, default=None)
    _add_common_args(p_exp)
    p_exp.set_defaults(func=cmd_export_annotation)

    p_imp = sub.add_parser("import-annotation", help="Import human annotation CSV and write back human_evaluation")
    p_imp.add_argument("--input", required=True, type=str, help="Annotations CSV path")
    p_imp.add_argument("--results", type=str, default="./data", help="Results root (per-model subfolders).")
    p_imp.add_argument("--out", type=str, default=None, help="Output root (default: overwrite in place)")
    _add_common_args(p_imp)
    p_imp.set_defaults(func=cmd_import_annotation)

    p_lb = sub.add_parser("leaderboard", help="Generate leaderboard with Model + 7 metrics columns")
    p_lb.add_argument("--input", required=True, type=str)
    p_lb.add_argument("--out", type=str, default=None,
                      help="Output directory (default: <input>/_leaderboard/)")
    _add_common_args(p_lb)
    p_lb.add_argument(
        "--metric-source",
        type=str,
        default="llm",
        choices=["llm", "automatic", "human"],
        dest="metric_source",
        help="Subjective metrics source (default llm).",
    )
    p_lb.set_defaults(func=cmd_leaderboard)

    p_viz = sub.add_parser(
        "visualize",
        help="Write metrics_radar.png (multi-model radar) from judged dialog records.",
    )
    p_viz.add_argument("--input", required=True, type=str,
                       help="Directory with per-model judged dialog records (e.g. data/_judged).")
    p_viz.add_argument("--out", type=str, default=None,
                       help="Output figures directory (default: <input>/_figures/).")
    _add_common_args(p_viz)
    p_viz.add_argument(
        "--metric-source",
        type=str,
        default="llm",
        choices=["llm", "automatic", "human"],
        dest="metric_source",
        help="Subjective metrics source (default llm).",
    )
    p_viz.set_defaults(func=cmd_visualize)

    p_ft = sub.add_parser(
        "prepare-judge-training",
        help="Build OpenAI fine-tuning JSONL from annotated/judged dialogs.",
    )
    p_ft.add_argument("--input", required=True, type=str,
                      help="Results root whose dialogs already carry human or llm scores.")
    p_ft.add_argument("--out", required=True, type=str, help="Output JSONL path (chat fine-tuning format).")
    p_ft.add_argument("--source", type=str, default="human", choices=["human", "llm"],
                      help="Where the labels come from (default human).")
    p_ft.add_argument("--val-fraction", type=float, default=0.0, dest="val_fraction",
                      help="Validation hold-out fraction (0 disables).")
    p_ft.add_argument("--val-out", type=str, default=None, dest="val_out")
    p_ft.add_argument("--prompt-version", type=str, default="v1.0.0", dest="prompt_version")
    p_ft.add_argument("--seed", type=int, default=1337)
    _add_common_args(p_ft)
    p_ft.set_defaults(func=cmd_prepare_judge_training)

    p_run = sub.add_parser(
        "run-all",
        help=(
            "One command: parse DESIGN.md -> dialog_specs.json, run llm-judge over "
            "all rollouts under data/<model>/, write the final leaderboard."
        ),
    )
    p_run.add_argument("--input", type=str, default=str(DEFAULT_DATA_ROOT),
                       help="Data root containing DESIGN.md and per-model rollout folders (default ./data).")
    p_run.add_argument("--design", type=str, default=None,
                       help="Override DESIGN.md path (default <input>/DESIGN.md).")
    p_run.add_argument("--specs", type=str, default=None,
                       help="Override dialog_specs.json output path (default <input>/dialog_specs.json).")
    p_run.add_argument("--judged", type=str, default=None,
                       help="Override judged-output dir (default <input>/_judged).")
    p_run.add_argument("--leaderboard", type=str, default=None,
                       help="Override leaderboard output dir (default <input>/_leaderboard).")
    p_run.add_argument("--figures", type=str, default=None,
                       help="Override figures output dir (default <leaderboard>/figures).")
    p_run.add_argument("--no-figures", action="store_true", dest="no_figures",
                       help="Skip the visualize step (e.g. when matplotlib is not installed).")
    p_run.add_argument(
        "--judge",
        type=str,
        default="openai",
        choices=["mock", "openai"],
        help="openai: real judge (default). mock: offline, no OPENAI_API_KEY.",
    )
    p_run.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OPENAI_JUDGE_MODEL,
        help=(
            "Model id passed to OpenAIJudge (default %(default)s). Ignored when --judge mock."
        ),
    )
    p_run.add_argument(
        "--blind-model-name",
        dest="blind_model_name",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Same as llm-judge; default hides rollout folder names from the judge.",
    )
    p_run.add_argument("--runs", type=int, default=1, help="Judge runs to average (default 1).")
    p_run.add_argument(
        "--annotated-design",
        type=str,
        default=None,
        help="Path to DESIGN_annotated.md (default: <DESIGN parent>/DESIGN_annotated.md if present).",
    )
    p_run.add_argument(
        "--no-annotated-overlay",
        action="store_true",
        dest="no_annotated_overlay",
        help="Do not merge DESIGN_annotated ground truth even if the file exists.",
    )
    p_run.set_defaults(func=cmd_run_all)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

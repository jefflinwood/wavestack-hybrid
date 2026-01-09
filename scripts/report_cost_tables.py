#!/usr/bin/env python
"""Generate cost tables (params/FLOPs/runtime) from EXPERIMENT_LOG.md."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re


def _parse_log(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    entry: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", line):
            if entry:
                entries.append(entry)
                entry = {}
            entry["timestamp"] = line.strip()
            continue
        if line.startswith("- "):
            key, _, value = line[2:].partition(":")
            entry[key.strip()] = value.strip()
    if entry:
        entries.append(entry)
    return entries


def _to_float(value: str | None) -> float | None:
    if value is None or value == "n/a":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    if value is None or value == "n/a":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_dict(value: str | None) -> dict[str, float]:
    if not value or value == "n/a":
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _latest_by_experiment(entries: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for entry in entries:
        name = entry.get("Experiment")
        if not name:
            continue
        if name not in latest or entry.get("timestamp", "") > latest[name].get("timestamp", ""):
            latest[name] = entry
    return latest


def _format_int(value: int | None) -> str:
    return f"{value:,}" if value is not None else "n/a"


def _format_float(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def _format_sci(value: float | None) -> str:
    return f"{value:.2e}" if value is not None else "n/a"


def _build_table(rows: list[dict[str, str]]) -> str:
    header = "| Experiment | Dataset | Params | FLOPs/seq | Tokens/s | Peak Mem (MB) |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        params_breakdown = _parse_dict(row.get("Params breakdown"))
        params = _to_int(row.get("Params total"))
        if not params or params == 0:
            if params_breakdown:
                params = int(sum(params_breakdown.values()))
        flops = _to_float(row.get("FLOPs total (seq)"))
        if flops is None or flops == 0.0:
            flops_breakdown = _parse_dict(row.get("FLOPs breakdown (seq)"))
            if flops_breakdown:
                flops = float(sum(flops_breakdown.values()))
        tokens_s = _to_float(row.get("Tokens/s"))
        mem_bytes = _to_int(row.get("Peak memory (bytes)"))
        mem_mb = mem_bytes / (1024 * 1024) if mem_bytes else None
        lines.append(
            "| {exp} | {dataset} | {params} | {flops} | {tps} | {mem} |".format(
                exp=row.get("Experiment", "n/a"),
                dataset=row.get("Dataset", "n/a"),
                params=_format_int(params),
                flops=_format_sci(flops),
                tps=_format_float(tokens_s, 2),
                mem=_format_float(mem_mb, 1),
            )
        )
    return "\n".join(lines)


def _filter_entries(entries: list[dict[str, str]], dataset: str | None) -> list[dict[str, str]]:
    filtered = entries
    if dataset:
        filtered = [e for e in entries if e.get("Dataset") == dataset]
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cost tables from EXPERIMENT_LOG.md.")
    parser.add_argument("--log", default="EXPERIMENT_LOG.md")
    parser.add_argument("--output", default="COSTS.md")
    parser.add_argument("--dataset", default=None, help="Filter to a dataset name.")
    args = parser.parse_args()

    entries = _parse_log(Path(args.log))
    latest = _latest_by_experiment(entries)
    rows = list(_filter_entries(list(latest.values()), args.dataset))
    rows = [row for row in rows if _parse_dict(row.get("Params breakdown"))]
    rows.sort(key=lambda r: (r.get("Dataset", ""), r.get("Experiment", "")))

    content = ["# Cost Tables", "", _build_table(rows), ""]
    Path(args.output).write_text("\n".join(content), encoding="utf-8")


if __name__ == "__main__":
    main()

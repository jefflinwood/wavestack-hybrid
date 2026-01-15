#!/usr/bin/env python
"""Generate matched hybrid vs transformer cost tables from EXPERIMENT_LOG.md."""

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


def _latest(entries: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for entry in entries:
        name = entry.get("Experiment") or entry.get("Model")
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


def _training_table(rows: list[dict[str, str]]) -> str:
    header = "| Experiment | Dataset | Params | FLOPs/seq | Tokens/s | Runtime (s) | Peak Mem (MB) |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        params = _to_int(row.get("Params total"))
        if not params or params == 0:
            params = int(sum(_parse_dict(row.get("Params breakdown")).values()) or 0) or None
        flops = _to_float(row.get("FLOPs total (seq)"))
        if flops is None or flops == 0.0:
            flops = float(sum(_parse_dict(row.get("FLOPs breakdown (seq)")).values()) or 0.0) or None
        tokens_s = _to_float(row.get("Tokens/s"))
        runtime_s = _to_float(row.get("Runtime (s)"))
        mem_bytes = _to_int(row.get("Peak memory (bytes)"))
        mem_mb = mem_bytes / (1024 * 1024) if mem_bytes else None
        lines.append(
            "| {exp} | {dataset} | {params} | {flops} | {tps} | {runtime} | {mem} |".format(
                exp=row.get("Experiment", "n/a"),
                dataset=row.get("Dataset", "n/a"),
                params=_format_int(params),
                flops=_format_sci(flops),
                tps=_format_float(tokens_s, 2),
                runtime=_format_float(runtime_s, 2),
                mem=_format_float(mem_mb, 1),
            )
        )
    return "\n".join(lines)


def _inference_table(rows: list[dict[str, str]]) -> str:
    header = "| Model | Device | Seq lens | Scaling exp | Timing |"
    sep = "| --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {model} | {device} | {seq} | {exp} | {timing} |".format(
                model=row.get("Model", "n/a"),
                device=row.get("Device", "n/a"),
                seq=row.get("Seq lens", "n/a"),
                exp=row.get("Scaling exponent", "n/a"),
                timing=row.get("Timing", "n/a"),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched cost tables.")
    parser.add_argument("--log", default="EXPERIMENT_LOG.md")
    parser.add_argument("--training-output", default="COSTS_TRAINING.md")
    parser.add_argument("--inference-output", default="COSTS_INFERENCE.md")
    args = parser.parse_args()

    entries = _parse_log(Path(args.log))
    latest = _latest(entries)

    training_names = {"hybrid_12m", "transformer_12m", "hybrid_50m", "transformer_50m"}
    training_rows = [
        latest[name]
        for name in training_names
        if name in latest and latest[name].get("Study") == "exp1_expressivity"
    ]
    training_rows.sort(key=lambda r: r.get("Experiment", ""))
    Path(args.training_output).write_text(
        "# Training Cost Tables\n\n" + _training_table(training_rows) + "\n",
        encoding="utf-8",
    )

    inference_rows = [
        entry
        for entry in entries
        if entry.get("Study") == "inference_benchmark"
        and entry.get("Model") in {"hybrid", "transformer"}
    ]
    inference_latest = list(_latest(inference_rows).values())
    inference_latest.sort(key=lambda r: r.get("Model", ""))
    Path(args.inference_output).write_text(
        "# Inference Cost Tables\n\n" + _inference_table(inference_latest) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

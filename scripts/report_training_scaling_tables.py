#!/usr/bin/env python
"""Generate training scaling tables from EXPERIMENT_LOG.md."""

from __future__ import annotations

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


def main() -> None:
    rows = [e for e in _parse_log(Path("EXPERIMENT_LOG.md")) if e.get("Study") == "training_benchmark"]
    latest = {}
    for row in rows:
        key = (row.get("Model"), row.get("Device"), row.get("Seq lens"))
        if key not in latest or row.get("timestamp", "") > latest[key].get("timestamp", ""):
            latest[key] = row
    lines = [
        "# Training Scaling Tables",
        "",
        "| Model | Device | Seq lens | Timing |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in sorted(latest.items(), key=lambda kv: (kv[0][1] or '', kv[0][0] or '')):
        lines.append(
            f"| {row.get('Model','n/a')} | {row.get('Device','n/a')} | {row.get('Seq lens','n/a')} | {row.get('Timing','n/a')} |"
        )
    Path("COSTS_TRAINING_SCALING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

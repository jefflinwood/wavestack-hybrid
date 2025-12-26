"""Simple metric aggregation utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict


class MetricTracker:
    """Keeps running averages for logging."""

    def __init__(self):
        self.storage: Dict[str, list[float]] = defaultdict(list)

    def update(self, key: str, value: float):
        self.storage[key].append(float(value))

    def compute(self) -> Dict[str, float]:
        return {key: sum(values) / max(1, len(values)) for key, values in self.storage.items()}

    def reset(self):
        self.storage.clear()

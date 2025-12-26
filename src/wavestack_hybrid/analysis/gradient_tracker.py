"""Utility for tracking gradient statistics during training."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch


class GradientTracker:
    """Collects gradient norms for critical subsystems."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.history: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    def log_gradients(self, step: int):
        if hasattr(self.model, "embeddings") and self.model.embeddings.token_embed.weight.grad is not None:
            grad = self.model.embeddings.token_embed.weight.grad
            self.history["embeddings_norm"].append({"step": step, "value": grad.norm().item()})

        lanes = getattr(self.model, "recomposition", None)
        if lanes is None:
            return

        for lane_name in ["poly", "trig", "wavelet"]:
            recomposition = lanes.lanes.get(lane_name) if hasattr(lanes, "lanes") else None
            if recomposition:
                grads = [p.grad.norm().item() for p in recomposition.parameters() if p.grad is not None]
                if grads:
                    self.history[f"{lane_name}_recomp"].append({"step": step, "value": float(np.mean(grads))})

    def summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for key, entries in self.history.items():
            values = [entry["value"] for entry in entries]
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "final": values[-1],
            }
        return summary

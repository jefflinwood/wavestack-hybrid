"""Visualization helpers for coefficients and gradients."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_lane_contributions(contributions: Dict[str, np.ndarray], title: str = "Lane contributions"):
    """Render stacked area plot summarizing lane strengths."""

    tokens = np.arange(next(iter(contributions.values())).shape[-1])
    data = np.stack(list(contributions.values()))
    labels = list(contributions.keys())

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.stackplot(tokens, data, labels=labels)
    ax.set_title(title)
    ax.set_xlabel("Token index")
    ax.set_ylabel("Contribution")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

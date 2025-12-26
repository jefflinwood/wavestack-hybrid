"""Simple training loop for experimenting locally."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import nn, optim

from ..config import ExperimentConfig
from .loss import compute_multi_objective_loss
from .metrics import MetricTracker


def _resolve_device(preference: str) -> torch.device:
    """Select an available device given user preference."""

    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(preference)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA selected but torch.cuda.is_available() is False.")

    if device.type == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise ValueError("MPS selected but torch.backends.mps.is_available() is False.")

    return device


class Trainer:
    """Minimal trainer supporting gradient accumulation and checkpointing."""

    def __init__(self, model: nn.Module, experiment: ExperimentConfig):
        self.model = model
        self.experiment = experiment
        self.device = _resolve_device(experiment.training.device)
        self.model.to(self.device)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=experiment.training.learning_rate,
            weight_decay=experiment.training.weight_decay,
        )
        self.use_amp = experiment.training.mixed_precision and self.device.type == "cuda"
        self.grad_scaler = (
            torch.amp.GradScaler(device_type="cuda", enabled=True) if self.use_amp else None
        )
        self.metric_tracker = MetricTracker()

    def train(self, dataloader: Iterable[Mapping[str, torch.Tensor]]):
        step = 0
        self.model.train()
        for batch in dataloader:
            loss = self._training_step(batch)
            loss_value = loss.item()
            self.metric_tracker.update("loss", loss_value)

            if (step + 1) % self.experiment.training.log_interval == 0:
                print(f"[step={step+1}] loss={self.metric_tracker.compute()['loss']:.4f}")
                self.metric_tracker.reset()

            if (step + 1) % self.experiment.training.save_interval == 0:
                self.save_checkpoint(step + 1)

            step += 1
            if step >= self.experiment.training.max_steps:
                break

    def _training_step(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=True) if self.use_amp else nullcontext()
        )
        with autocast_ctx:
            logits = self.model(input_ids)
            loss = compute_multi_objective_loss(logits, labels, None, None, self.experiment.training)

        self.optimizer.zero_grad()
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.experiment.training.max_grad_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.experiment.training.max_grad_norm)
            self.optimizer.step()

        return loss.detach()

    def save_checkpoint(self, step: int):
        output_dir = Path(self.experiment.checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"checkpoint_{step:06d}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": step,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")

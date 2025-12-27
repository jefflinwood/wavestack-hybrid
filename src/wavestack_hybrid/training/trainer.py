"""Simple training loop for experimenting locally."""

from __future__ import annotations

from contextlib import nullcontext
import json
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
        self.output_dir = Path(self.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / f"{self.experiment.name}_metrics.jsonl"
        self.last_train_loss: float | None = None
        self.last_eval_loss: float | None = None

    def train(
        self,
        dataloader: Iterable[Mapping[str, torch.Tensor]],
        eval_dataloader: Iterable[Mapping[str, torch.Tensor]] | None = None,
    ) -> Mapping[str, float | int | None]:
        step = 0
        self.model.train()
        if hasattr(dataloader, "__len__") and len(dataloader) == 0:
            raise ValueError("Dataloader is empty; no training steps to run.")

        while step < self.experiment.training.max_steps:
            for batch in dataloader:
                loss = self._training_step(batch)
                loss_value = loss.item()
                self.metric_tracker.update("loss", loss_value)

                train_loss_avg = None
                eval_loss = None

                if (
                    eval_dataloader is not None
                    and (step + 1) % self.experiment.training.eval_interval == 0
                ):
                    eval_loss = self.evaluate(eval_dataloader)
                    self.last_eval_loss = eval_loss
                    print(f"[step={step+1}] eval_loss={eval_loss:.4f}")

                if (step + 1) % self.experiment.training.log_interval == 0:
                    train_loss_avg = self.metric_tracker.compute()["loss"]
                    self.last_train_loss = train_loss_avg
                    print(f"[step={step+1}] loss={train_loss_avg:.4f}")
                    self.metric_tracker.reset()

                if train_loss_avg is not None or eval_loss is not None:
                    payload: dict[str, float | int] = {"step": step + 1}
                    if train_loss_avg is not None:
                        payload["train_loss"] = train_loss_avg
                    if eval_loss is not None:
                        payload["eval_loss"] = eval_loss
                    self._append_log(payload)

                if (step + 1) % self.experiment.training.save_interval == 0:
                    self.save_checkpoint(step + 1)

                step += 1
                if step >= self.experiment.training.max_steps:
                    break

        if self.metric_tracker.storage:
            final_train_loss = self.metric_tracker.compute().get("loss")
            if final_train_loss is not None:
                self.last_train_loss = final_train_loss
                self._append_log({"step": step, "train_loss": final_train_loss})
            self.metric_tracker.reset()

        return {
            "steps": step,
            "train_loss": self.last_train_loss,
            "eval_loss": self.last_eval_loss,
        }

    def evaluate(self, dataloader: Iterable[Mapping[str, torch.Tensor]]) -> float:
        max_batches = self.experiment.training.eval_batches
        total_loss = 0.0
        batches = 0
        was_training = self.model.training
        self.model.eval()
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=True) if self.use_amp else nullcontext()
        )
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                with autocast_ctx:
                    logits = self.model(input_ids)
                    loss = compute_multi_objective_loss(
                        logits, labels, None, None, self.experiment.training
                    )
                total_loss += float(loss.item())
                batches += 1
                if batches >= max_batches:
                    break
        if was_training:
            self.model.train()
        if batches == 0:
            raise ValueError("Eval dataloader is empty; no evaluation steps to run.")
        return total_loss / batches

    def _append_log(self, payload: Mapping[str, float | int]):
        with self.metrics_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")

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

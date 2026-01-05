"""Simple training loop for experimenting locally."""

from __future__ import annotations

from contextlib import nullcontext
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import nn, optim
import torch.nn.functional as F

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
        self.last_lane_stats: Mapping[str, float] | None = None

    def train(
        self,
        dataloader: Iterable[Mapping[str, torch.Tensor]],
        eval_dataloader: Iterable[Mapping[str, torch.Tensor]] | None = None,
    ) -> Mapping[str, float | int | None]:
        step = 0
        if accumulation_steps != 1:
            print(
                "[Trainer] Gradient accumulation steps="
                f"{accumulation_steps} (effective batch size="
                f"{self.experiment.training.batch_size * accumulation_steps})"
            )
        accumulation_steps = max(1, self.experiment.training.gradient_accumulation_steps)
        accumulation_count = 0
        interval_steps = 0
        interval_time = 0.0
        interval_tokens = 0
        total_time = 0.0
        total_tokens = 0
        peak_memory_bytes: int | None = None
        self.model.train()
        if hasattr(dataloader, "__len__") and len(dataloader) == 0:
            raise ValueError("Dataloader is empty; no training steps to run.")
        if self.experiment.training.log_memory and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        self.optimizer.zero_grad()
        while step < self.experiment.training.max_steps:
            for batch in dataloader:
                if hasattr(self.model, "update_schedule"):
                    self.model.update_schedule(step + 1, self.experiment.training.max_steps)
                start_time = time.perf_counter()
                loss = self._compute_loss(batch)
                loss_to_backprop = loss / accumulation_steps
                if self.grad_scaler is not None:
                    self.grad_scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()
                step_time = time.perf_counter() - start_time
                batch_tokens = int(batch["input_ids"].numel())
                total_time += step_time
                total_tokens += batch_tokens
                interval_time += step_time
                interval_tokens += batch_tokens
                interval_steps += 1
                if self.experiment.training.log_memory:
                    memory = self._get_memory_bytes()
                    if memory is not None:
                        peak_memory_bytes = memory if peak_memory_bytes is None else max(
                            peak_memory_bytes, memory
                        )
                loss_value = loss.item()
                self.metric_tracker.update("loss", loss_value)
                accumulation_count += 1

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
                    if self.experiment.training.log_runtime and interval_steps > 0:
                        avg_step = interval_time / interval_steps
                        payload["step_time_ms"] = avg_step * 1000.0
                        payload["tokens_per_s"] = interval_tokens / max(1e-8, interval_time)
                    if self.experiment.training.log_memory and peak_memory_bytes is not None:
                        payload["peak_memory_bytes"] = peak_memory_bytes
                    if self.last_lane_stats:
                        payload.update(self.last_lane_stats)
                    self._append_log(payload)
                    interval_steps = 0
                    interval_time = 0.0
                    interval_tokens = 0

                if accumulation_count >= accumulation_steps:
                    if self.grad_scaler is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.experiment.training.max_grad_norm
                        )
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.experiment.training.max_grad_norm
                        )
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulation_count = 0

                if (step + 1) % self.experiment.training.save_interval == 0:
                    self.save_checkpoint(step + 1)

                step += 1
                if step >= self.experiment.training.max_steps:
                    break

        if accumulation_count > 0:
            if self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.experiment.training.max_grad_norm
                )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.experiment.training.max_grad_norm
                )
                self.optimizer.step()
            self.optimizer.zero_grad()
            accumulation_count = 0

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
            "runtime_s": total_time if total_time else None,
            "tokens_per_s": (total_tokens / total_time) if total_time else None,
            "peak_memory_bytes": peak_memory_bytes,
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

    def _compute_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        self.last_lane_stats = None

        want_lanes = self.experiment.training.lane_diversity or self.experiment.training.log_lane_stats

        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=True) if self.use_amp else nullcontext()
        )
        with autocast_ctx:
            if want_lanes:
                logits, lane_outputs = self.model(input_ids, return_lanes=True)
                lane_balance = self._compute_lane_balance(lane_outputs)
                loss = compute_multi_objective_loss(
                    logits, labels, None, lane_balance, self.experiment.training
                )
                if self.experiment.training.log_lane_stats:
                    self.last_lane_stats = self._compute_lane_stats(lane_outputs)
            else:
                logits = self.model(input_ids)
                loss = compute_multi_objective_loss(
                    logits, labels, None, None, self.experiment.training
                )

        return loss.detach()

    def _compute_lane_balance(self, lane_outputs: torch.Tensor) -> torch.Tensor:
        if not self.experiment.training.lane_diversity:
            return torch.tensor(0.0, device=lane_outputs.device)
        metric = self.experiment.training.lane_diversity_metric
        if lane_outputs.size(0) < 2:
            return torch.tensor(0.0, device=lane_outputs.device)
        if metric == "energy":
            energies = lane_outputs.pow(2).mean(dim=(1, 2, 3))
            mean = energies.mean().clamp_min(1e-6)
            return energies.std() / mean
        normalized = F.normalize(lane_outputs, dim=-1, eps=1e-6)
        sims = []
        for i in range(normalized.size(0)):
            for j in range(i + 1, normalized.size(0)):
                sims.append((normalized[i] * normalized[j]).sum(dim=-1).mean())
        if not sims:
            return torch.tensor(0.0, device=lane_outputs.device)
        return torch.stack(sims).mean()

    def _compute_lane_stats(self, lane_outputs: torch.Tensor) -> Mapping[str, float]:
        names = getattr(self.model, "lane_names", None)
        if not names:
            names = [f"lane_{idx}" for idx in range(lane_outputs.size(0))]
        norms = torch.linalg.vector_norm(lane_outputs, dim=-1).mean(dim=(1, 2))
        return {f"{name}_norm": float(norm.item()) for name, norm in zip(names, norms)}

    def _get_memory_bytes(self) -> int | None:
        if self.device.type == "cuda":
            return int(torch.cuda.max_memory_allocated(self.device))
        if self.device.type == "mps":
            current = getattr(torch.mps, "current_allocated_memory", None)
            if current is not None:
                return int(current())
            return None
        if self.device.type == "cpu":
            try:
                import resource

                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform == "darwin":
                    return int(rss)
                return int(rss * 1024)
            except Exception:
                return None
        return None

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

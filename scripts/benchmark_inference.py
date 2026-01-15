#!/usr/bin/env python
"""Benchmark inference throughput vs sequence length for Hybrid and Transformer baselines."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.models.wavestack import HybridWaveStack


@dataclass
class BenchmarkResult:
    model_name: str
    seq_len: int
    batch_size: int
    time_s: float
    tokens_per_s: float
    memory_bytes: int | None


class TransformerBlock(nn.Module):
    """Minimal causal transformer block for throughput baselines."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Simple GPT-style LM for scaling comparisons."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token_embed(input_ids) + self.pos_embed(positions)
        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        hidden = self.ln(hidden)
        return self.head(hidden)


def _parse_seq_lens(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()


def _get_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps":
        current = getattr(torch.mps, "current_allocated_memory", None)
        if current is not None:
            return int(current())
    return None


def _fit_power(xs: Iterable[int], ys: Iterable[float]) -> float:
    xs = list(xs)
    ys = list(ys)
    logx = [math.log(x) for x in xs]
    logy = [math.log(y) for y in ys]
    mean_x = sum(logx) / len(logx)
    mean_y = sum(logy) / len(logy)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(logx, logy))
    var = sum((x - mean_x) ** 2 for x in logx)
    return cov / var if var else 0.0


def _benchmark_model(
    model: nn.Module,
    device: torch.device,
    seq_lens: list[int],
    batch_size: int,
    steps: int,
    warmup: int,
    vocab_size: int,
    max_seq_len: int,
    model_name: str,
    output_path: Path | None,
    use_attention_mask: bool = False,
    log_summary: bool = False,
) -> list[BenchmarkResult]:
    model.eval()
    model.to(device)
    results: list[BenchmarkResult] = []
    for seq_len in seq_lens:
        if seq_len > max_seq_len:
            print(f"[benchmark] Skipping seq_len={seq_len} (max_seq_len={max_seq_len})")
            continue
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attn_mask = None
        if use_attention_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            attn_mask = mask
        with torch.no_grad():
            for _ in range(warmup):
                if use_attention_mask:
                    _ = model(input_ids, attn_mask)
                else:
                    _ = model(input_ids)
            _sync(device)
            start = time.perf_counter()
            for _ in range(steps):
                if use_attention_mask:
                    _ = model(input_ids, attn_mask)
                else:
                    _ = model(input_ids)
            _sync(device)
            total_time = time.perf_counter() - start
        memory_bytes = _get_memory_bytes(device)
        time_per_step = total_time / max(1, steps)
        tokens_per_s = (batch_size * seq_len) / time_per_step
        result = BenchmarkResult(
            model_name=model_name,
            seq_len=seq_len,
            batch_size=batch_size,
            time_s=time_per_step,
            tokens_per_s=tokens_per_s,
            memory_bytes=memory_bytes,
        )
        results.append(result)
        if output_path:
            with output_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(result.__dict__) + "\n")
    return results


def _print_summary(results: list[BenchmarkResult]) -> float:
    times = [res.time_s for res in results]
    seqs = [res.seq_len for res in results]
    exponent = _fit_power(seqs, times)
    print(f"\n{results[0].model_name} scaling: time ~ seq_len^{exponent:.2f}")
    print("seq_len  time_ms  tokens_per_s  memory_bytes")
    for res in results:
        mem_str = f"{res.memory_bytes}" if res.memory_bytes is not None else "n/a"
        print(
            f"{res.seq_len:>7}  {res.time_s*1000:>7.2f}  {res.tokens_per_s:>12.1f}  {mem_str:>12}"
        )
    return exponent


def _append_experiment_log(
    model_name: str,
    results: list[BenchmarkResult],
    batch_size: int,
    steps: int,
    warmup: int,
    device: str,
    exponent: float,
    output_path: Path | None,
) -> None:
    seq_lens = [str(r.seq_len) for r in results]
    timing_pairs = ", ".join(
        f"{r.seq_len}:{r.time_s*1000:.2f}ms/{r.tokens_per_s:.1f}tps/{r.memory_bytes or 0}B"
        for r in results
    )
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        "",
        timestamp,
        "- Study: inference_benchmark",
        f"- Model: {model_name}",
        f"- Device: {device}",
        f"- Seq lens: {','.join(seq_lens)}",
        f"- Batch size: {batch_size}",
        f"- Steps: {steps}",
        f"- Warmup: {warmup}",
        f"- Scaling exponent: {exponent:.3f}",
        f"- Timing: {timing_pairs}",
        f"- Output: {output_path}" if output_path else "- Output: n/a",
    ]
    log_path = Path("EXPERIMENT_LOG.md")
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference scaling benchmark.")
    parser.add_argument("--model", default="both", choices=["hybrid", "transformer", "both"])
    parser.add_argument("--config", default=None, help="Hybrid config YAML path.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path for hybrid.")
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps.")
    parser.add_argument("--seq-lens", default="64,128,256,512", help="Comma-separated seq lengths.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--tf-hidden-dim", type=int, default=512)
    parser.add_argument("--tf-layers", type=int, default=6)
    parser.add_argument("--tf-heads", type=int, default=8)
    parser.add_argument("--tf-vocab-size", type=int, default=50_257)
    parser.add_argument("--tf-max-seq-len", type=int, default=512)
    parser.add_argument(
        "--log-results",
        action="store_true",
        help="Append benchmark summary to EXPERIMENT_LOG.md.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
    else:
        device = torch.device(args.device)

    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    seq_lens = _parse_seq_lens(args.seq_lens)

    if args.model in {"hybrid", "both"}:
        if not args.config:
            raise ValueError("--config is required for hybrid benchmarks.")
        experiment = ExperimentConfig.from_yaml(args.config)
        model = HybridWaveStack(experiment.model)
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state["model"] if isinstance(state, dict) else state)
        results = _benchmark_model(
            model=model,
            device=device,
            seq_lens=seq_lens,
            batch_size=args.batch_size,
            steps=args.steps,
            warmup=args.warmup,
            vocab_size=experiment.model.vocab_size,
            max_seq_len=experiment.model.max_seq_len,
            model_name="hybrid",
            output_path=output_path,
            use_attention_mask=False,
        )
        exponent = _print_summary(results)
        if args.log_results:
            _append_experiment_log(
                model_name="hybrid",
                results=results,
                batch_size=args.batch_size,
                steps=args.steps,
                warmup=args.warmup,
                device=str(device),
                exponent=exponent,
                output_path=output_path,
            )

    if args.model in {"transformer", "both"}:
        model = TransformerLM(
            vocab_size=args.tf_vocab_size,
            hidden_dim=args.tf_hidden_dim,
            num_layers=args.tf_layers,
            num_heads=args.tf_heads,
            max_seq_len=args.tf_max_seq_len,
        )
        results = _benchmark_model(
            model=model,
            device=device,
            seq_lens=seq_lens,
            batch_size=args.batch_size,
            steps=args.steps,
            warmup=args.warmup,
            vocab_size=args.tf_vocab_size,
            max_seq_len=args.tf_max_seq_len,
            model_name="transformer",
            output_path=output_path,
            use_attention_mask=True,
        )
        exponent = _print_summary(results)
        if args.log_results:
            _append_experiment_log(
                model_name="transformer",
                results=results,
                batch_size=args.batch_size,
                steps=args.steps,
                warmup=args.warmup,
                device=str(device),
                exponent=exponent,
                output_path=output_path,
            )


if __name__ == "__main__":
    main()

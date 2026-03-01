#!/usr/bin/env python3
"""
Spike Verification at 30% / 70% SM — OPT-1.3b Decode Matrix Shapes
====================================================================

Hypothesis: the decode latency spikes at 30% and 70% MPS SM allocation
are caused by cuBLAS selecting a suboptimal GEMM kernel at those SM counts.

Approach:
  Decompose OPT-1.3b decode into its individual matrix operations and
  time each one independently across all SM percentages. If the spike
  appears only in certain op types (e.g. FFN linear but not attention),
  that isolates the root cause.

OPT-1.3b decode ops per layer (batch=1, seq_step=1):
  ① Linear (square)    : (1,2048) × (2048,2048)  — Q/K/V/O projections
  ② Linear (FFN up)    : (1,2048) × (2048,8192)  — FFN expand
  ③ Linear (FFN down)  : (1,8192) × (8192,2048)  — FFN contract
  ④ Attn score (bmm)   : (32,1,64) × (32,64,L)   — Q @ K^T  (L=KV len)
  ⑤ Attn context (bmm) : (32,1,L)  × (32,L,64)   — scores @ V

Usage:
  # full sweep (spawns subprocesses via MPS)
  python verify_spike.py

  # re-plot from existing JSON
  python verify_spike.py --mode plot
"""

import os, sys, json, time, argparse, subprocess
from pathlib import Path
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────
GPU_PERCENTAGES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
KV_LEN          = 512       # KV-cache length for attention ops (matches decode step 512)
WARMUP_RUNS     = 5
MEASURE_RUNS    = 20        # more runs to reduce noise for these tiny ops
RESULTS_FILE    = Path(__file__).parent / "spike_results.json"
PLOT_FILE       = Path(__file__).parent / "spike_result.png"

# OPT-1.3b decode matrix shapes
OPS = {
    "linear_square" : dict(M=1, N=2048, K=2048,  desc="Q/K/V/O proj  (1×2048)×(2048×2048)"),
    "ffn_up"        : dict(M=1, N=8192, K=2048,  desc="FFN expand    (1×2048)×(2048×8192)"),
    "ffn_down"      : dict(M=1, N=2048, K=8192,  desc="FFN contract  (1×8192)×(8192×2048)"),
    "attn_score"    : dict(M=1, N=KV_LEN, K=64,  desc=f"Q@K^T bmm     (32,1,64)×(32,64,{KV_LEN})"),
    "attn_context"  : dict(M=1, N=64,    K=KV_LEN, desc=f"scores@V bmm  (32,1,{KV_LEN})×(32,{KV_LEN},64)"),
}


# ─── Worker ────────────────────────────────────────────────────────────────
def worker():
    import torch

    gpu_pct = int(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", 100))
    assert torch.cuda.is_available()
    dev = "cuda:0"

    def bench_gemm(M, N, K):
        """Time a single (M,N,K) GEMM in fp16."""
        A = torch.randn(M, K, device=dev, dtype=torch.float16)
        B = torch.randn(K, N, device=dev, dtype=torch.float16)
        # warm-up
        for _ in range(WARMUP_RUNS):
            torch.cuda.synchronize()
            _ = torch.mm(A, B)
        torch.cuda.synchronize()
        # measure
        times = []
        for _ in range(MEASURE_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.mm(A, B)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)   # µs
        return float(np.mean(times)), float(np.std(times))

    def bench_bmm(batch, M, N, K):
        """Time a batched (batch,M,K)×(batch,K,N) in fp16."""
        A = torch.randn(batch, M, K, device=dev, dtype=torch.float16)
        B = torch.randn(batch, K, N, device=dev, dtype=torch.float16)
        for _ in range(WARMUP_RUNS):
            torch.cuda.synchronize()
            _ = torch.bmm(A, B)
        torch.cuda.synchronize()
        times = []
        for _ in range(MEASURE_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.bmm(A, B)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)
        return float(np.mean(times)), float(np.std(times))

    row = {"gpu_pct": gpu_pct}
    for name, cfg in OPS.items():
        M, N, K = cfg["M"], cfg["N"], cfg["K"]
        if name.startswith("attn"):
            mean_us, std_us = bench_bmm(32, M, N, K)   # 32 heads
        else:
            mean_us, std_us = bench_gemm(M, N, K)
        row[f"{name}_us"]  = mean_us
        row[f"{name}_std"] = std_us
        print(f"[worker {gpu_pct}%] {name:16s}: {mean_us:7.2f} ± {std_us:.2f} µs",
              file=sys.stderr, flush=True)

    print(json.dumps(row), flush=True)


# ─── Orchestrator ──────────────────────────────────────────────────────────
def orchestrate():
    all_results = []
    total = len(GPU_PERCENTAGES)

    for idx, pct in enumerate(GPU_PERCENTAGES, 1):
        print(f"\n{'─'*50}  [{idx}/{total}]  SM={pct}%")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]             = "0"
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)

        proc = subprocess.run(
            [sys.executable, __file__, "--mode=worker"],
            env=env, capture_output=True, text=True, timeout=300,
        )
        for line in proc.stderr.strip().splitlines():
            print(f"  {line}")

        for line in proc.stdout.strip().splitlines():
            if line.startswith("{"):
                all_results.append(json.loads(line))
                break
        else:
            print(f"  [WARN] no result  rc={proc.returncode}")

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\n[Done] {len(all_results)} results → {RESULTS_FILE}")
    return all_results


# ─── Plot ──────────────────────────────────────────────────────────────────
def plot(results=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    if results is None:
        results = json.loads(RESULTS_FILE.read_text())

    pcts = sorted(r["gpu_pct"] for r in results)
    by_pct = {r["gpu_pct"]: r for r in results}

    op_names   = list(OPS.keys())
    op_labels  = {
        "linear_square" : "① Linear square\n(Q/K/V/O proj)",
        "ffn_up"        : "② FFN up\n(2048→8192)",
        "ffn_down"      : "③ FFN down\n(8192→2048)",
        "attn_score"    : "④ Attn score\n(Q @ K^T)",
        "attn_context"  : "⑤ Attn context\n(scores @ V)",
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # ── Main figure: one subplot per op ───────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
    fig.suptitle(
        "Spike Verification — Isolated GEMM/BMM ops at OPT-1.3b Decode Shapes\n"
        f"fp16, batch=1, {MEASURE_RUNS} runs per point (mean ± std)",
        fontsize=12, fontweight="bold"
    )

    for ax, name, color in zip(axes, op_names, colors):
        xs  = pcts
        ys  = [by_pct[p][f"{name}_us"]  for p in xs]
        es  = [by_pct[p][f"{name}_std"] for p in xs]

        ax.errorbar(xs, ys, yerr=es, color=color,
                    marker="o", linewidth=2.0, markersize=6,
                    capsize=4, capthick=1.5)

        # Highlight spike positions
        for spike_pct in [30, 70]:
            if spike_pct in by_pct:
                sy = by_pct[spike_pct][f"{name}_us"]
                ax.axvline(spike_pct, color="red", linestyle=":", linewidth=1.2, alpha=0.6)
                ax.scatter([spike_pct], [sy], color="red", zorder=5, s=60)

        ax.set_xlabel("GPU SM Utilization (%)", fontsize=10)
        ax.set_ylabel("Latency (µs)", fontsize=10)
        ax.set_title(op_labels[name], fontsize=10, fontweight="bold")
        ax.set_xticks(pcts)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {PLOT_FILE}")

    # ── Summary: normalized latency vs baseline (100%) ─────────────────
    print("\n" + "═"*90)
    base = by_pct.get(100, {})
    header = f"{'SM%':>4}  " + "  ".join(f"{n:>16}" for n in op_names)
    print(header)
    print("─"*90)
    for p in pcts:
        r = by_pct[p]
        vals = []
        for name in op_names:
            us  = r[f"{name}_us"]
            b   = base.get(f"{name}_us", us)
            norm = us / b if b else 1.0
            spike = " ←SPIKE" if abs(norm - 1.0) > 0.15 and p in [30, 70] else ""
            vals.append(f"{us:6.1f}µs({norm:.2f}x){spike}")
        print(f"{p:>3}%  " + "  ".join(f"{v:>16}" for v in vals))
    print("═"*90)
    print("(x value = normalized latency vs 100% SM baseline; >1.0 means slower)")


# ─── Entry ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="orchestrate",
                        choices=["orchestrate", "worker", "plot"])
    args = parser.parse_args()

    if args.mode == "orchestrate":
        results = orchestrate()
        plot(results)
    elif args.mode == "worker":
        worker()
    elif args.mode == "plot":
        plot()


if __name__ == "__main__":
    main()

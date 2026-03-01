#!/usr/bin/env python3
"""
Spike Verification Round 2 — Non-GEMM ops in OPT-1.3b Decode
=============================================================

Round 1 showed that isolated GEMM/BMM operations have NO spikes at 30%/70%
SM — they follow a smooth monotone curve. This falsifies the cuBLAS hypothesis.

The spike must come from non-GEMM ops. This script tests:
  ① LayerNorm          : F.layer_norm(x, [2048])
  ② Softmax            : F.softmax(attn_weights, dim=-1)  shape=(1,32,1,512)
  ③ Elementwise add    : residual stream addition, shape=(1,1,2048)
  ④ Combined elementwise (all 3 per layer × 24 layers simulation)
  ⑤ Full non-GEMM layer simulation (mimics one decode layer minus GEMM)

If the spike appears in ④ or ⑤ but not ①②③ individually,
the overhead is from kernel launch accumulation at specific SM counts.
"""

import os, sys, json, time, argparse, subprocess
from pathlib import Path
import numpy as np

GPU_PERCENTAGES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
WARMUP_RUNS     = 10
MEASURE_RUNS    = 100    # many tiny ops need more runs for stable measurement
NUM_LAYERS      = 24     # simulate 24 decoder layers
KV_LEN          = 512
RESULTS_FILE    = Path(__file__).parent / "spike2_results.json"
PLOT_FILE       = Path(__file__).parent / "spike2_result.png"

OPS = {
    "layernorm"       : "LayerNorm on (1,1,2048)",
    "softmax"         : f"Softmax on (1,32,1,{KV_LEN})",
    "elem_add"        : "Elementwise add (1,1,2048)",
    "layer_stack"     : f"24× (LN + add + softmax) — no GEMM",
}


def worker():
    import torch
    import torch.nn.functional as F

    gpu_pct = int(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", 100))
    assert torch.cuda.is_available()
    dev = "cuda:0"

    def bench(fn, setup_fn, n_warm=WARMUP_RUNS, n_meas=MEASURE_RUNS):
        setup_fn()  # create tensors
        for _ in range(n_warm):
            torch.cuda.synchronize()
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(n_meas):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)
        return float(np.mean(times)), float(np.std(times))

    results = {}

    # ① LayerNorm
    x_ln = [None]
    def setup_ln(): x_ln[0] = torch.randn(1, 1, 2048, device=dev, dtype=torch.float16)
    def run_ln():   F.layer_norm(x_ln[0], [2048])
    m, s = bench(run_ln, setup_ln)
    results["layernorm"] = (m, s)
    print(f"[{gpu_pct}%] layernorm   : {m:.3f} ± {s:.3f} µs", file=sys.stderr, flush=True)

    # ② Softmax
    attn = [None]
    def setup_sm(): attn[0] = torch.randn(1, 32, 1, KV_LEN, device=dev, dtype=torch.float16)
    def run_sm():   F.softmax(attn[0], dim=-1)
    m, s = bench(run_sm, setup_sm)
    results["softmax"] = (m, s)
    print(f"[{gpu_pct}%] softmax     : {m:.3f} ± {s:.3f} µs", file=sys.stderr, flush=True)

    # ③ Elementwise add
    a_e, b_e = [None], [None]
    def setup_add():
        a_e[0] = torch.randn(1, 1, 2048, device=dev, dtype=torch.float16)
        b_e[0] = torch.randn(1, 1, 2048, device=dev, dtype=torch.float16)
    def run_add(): torch.add(a_e[0], b_e[0])
    m, s = bench(run_add, setup_add)
    results["elem_add"] = (m, s)
    print(f"[{gpu_pct}%] elem_add    : {m:.3f} ± {s:.3f} µs", file=sys.stderr, flush=True)

    # ④ 24-layer stack of non-GEMM ops only
    # Mimics: LN1 + residual_add + QKV_LN + softmax + LN2 + residual_add  (per layer)
    def setup_stack():
        pass   # reuse tensors from above; call setups
        setup_ln(); setup_sm(); setup_add()
    def run_stack():
        for _ in range(NUM_LAYERS):
            F.layer_norm(x_ln[0], [2048])    # pre-attn LN
            torch.add(a_e[0], b_e[0])        # residual after attn
            F.softmax(attn[0], dim=-1)       # attention softmax
            F.layer_norm(x_ln[0], [2048])    # pre-FFN LN
            torch.add(a_e[0], b_e[0])        # residual after FFN
    setup_stack()
    m, s = bench(run_stack, setup_stack, n_warm=5, n_meas=50)
    results["layer_stack"] = (m, s)
    print(f"[{gpu_pct}%] layer_stack : {m:.3f} ± {s:.3f} µs  (24 layers)", file=sys.stderr, flush=True)

    row = {"gpu_pct": gpu_pct}
    for name, (mean_us, std_us) in results.items():
        row[f"{name}_us"]  = mean_us
        row[f"{name}_std"] = std_us
    print(json.dumps(row), flush=True)


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
            print(f"  [WARN] no result  rc={proc.returncode}\n{proc.stderr[-200:]}")

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\n[Done] {len(all_results)} results → {RESULTS_FILE}")
    return all_results


def plot(results=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if results is None:
        results = json.loads(RESULTS_FILE.read_text())

    pcts   = sorted(r["gpu_pct"] for r in results)
    by_pct = {r["gpu_pct"]: r for r in results}
    names  = list(OPS.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    labels = {
        "layernorm"   : "① LayerNorm\n(1,1,2048)",
        "softmax"     : f"② Softmax\n(1,32,1,{KV_LEN})",
        "elem_add"    : "③ Elementwise Add\n(1,1,2048)",
        "layer_stack" : "④ 24-Layer Non-GEMM Stack\n(LN+add+softmax) × 24",
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        "Spike Verification Round 2 — Non-GEMM Ops in OPT-1.3b Decode\n"
        f"{MEASURE_RUNS} runs per point (mean ± std); red lines at 30% and 70%",
        fontsize=12, fontweight="bold"
    )

    for ax, name, color in zip(axes, names, colors):
        ys  = [by_pct[p][f"{name}_us"]  for p in pcts]
        es  = [by_pct[p][f"{name}_std"] for p in pcts]
        base = by_pct[100][f"{name}_us"]

        ax.errorbar(pcts, ys, yerr=es, color=color,
                    marker="o", linewidth=2.0, markersize=6,
                    capsize=4, capthick=1.5)

        for spike_pct in [30, 70]:
            sy = by_pct[spike_pct][f"{name}_us"]
            ax.axvline(spike_pct, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
            ax.scatter([spike_pct], [sy], color="red", zorder=5, s=70)

        ax.set_xlabel("GPU SM Utilization (%)", fontsize=10)
        ax.set_ylabel("Latency (µs)", fontsize=10)
        ax.set_title(labels[name], fontsize=10, fontweight="bold")
        ax.set_xticks(pcts)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Annotate normalized values at 30% and 70%
        for p in [30, 70]:
            y = by_pct[p][f"{name}_us"]
            norm = y / base
            ax.annotate(
                f"{norm:.2f}x", xy=(p, y),
                xytext=(p + 3, y + (max(ys) - min(ys)) * 0.05),
                fontsize=7, color="red",
            )

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {PLOT_FILE}")

    # Summary table
    print("\n" + "═"*80)
    base_row = by_pct.get(100, {})
    print(f"{'SM%':>4}  " + "  ".join(f"{n:>18}" for n in names))
    print("─"*80)
    for p in pcts:
        r = by_pct[p]
        vals = []
        for name in names:
            us   = r[f"{name}_us"]
            b    = base_row.get(f"{name}_us", us)
            norm = us / b if b else 1.0
            flag = " ★" if (p in [30, 70] and norm > 1.15) else ""
            vals.append(f"{us:5.1f}µs({norm:.2f}x){flag}")
        print(f"{p:>3}%  " + "  ".join(f"{v:>18}" for v in vals))
    print("═"*80)
    print("★ = spike at 30%/70% relative to 100% baseline (>1.15x)")


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

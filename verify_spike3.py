#!/usr/bin/env python3
"""
Spike Verification Round 3 — Layer-Count Scaling Test
======================================================

Hypothesis: the decode spike at 30%/70% SM comes from MPS per-kernel-launch
overhead accumulating across ~240 sequential CUDA kernels (24 layers × ~10 ops).

Prediction: if the hypothesis is correct:
  extra_latency = per_kernel_overhead(SM%) × total_kernels
  → total latency at SM% scales LINEARLY with layer count
  → slope(30%) > slope(100%) and slope(70%) > slope(100%)
  → (slope(30%) - slope(100%)) × 24 layers ≈ 3.8 ms  (observed spike)

Experiment:
  For each (SM%, layer_count) pair, run a simulated decode stack with
  exactly `layer_count` layers (each layer = same ops as OPT-1.3b decode):
    GEMM square  → LayerNorm → residual add → GEMM ffn_up →
    GEMM ffn_down → attn score BMM → softmax → attn context BMM →
    GEMM output → LayerNorm → residual add
  That is 9 kernel launches per layer (plus extras = ~10 kernels/layer).

  Fit: time = a × layers + b
  Compare slope `a` across SM% values.
"""

import os, sys, json, time, argparse, subprocess
from pathlib import Path
import numpy as np

GPU_PERCENTAGES = [30, 40, 70, 80, 100]    # focus on spike vs neighbours
LAYER_COUNTS    = [1, 3, 6, 12, 24, 48]    # sweep layer count
KV_LEN          = 512
WARMUP_RUNS     = 3
MEASURE_RUNS    = 10
RESULTS_FILE    = Path(__file__).parent / "spike3_results.json"
PLOT_FILE       = Path(__file__).parent / "spike3_result.png"

# ─── Worker ────────────────────────────────────────────────────────────────
def worker():
    import torch
    import torch.nn.functional as F

    gpu_pct = int(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", 100))
    assert torch.cuda.is_available()
    dev = "cuda:0"

    # Pre-allocate all tensors once (no allocation overhead in the timed loop)
    W_sq  = torch.randn(2048, 2048, device=dev, dtype=torch.float16)  # Q/K/V/O
    W_up  = torch.randn(2048, 8192, device=dev, dtype=torch.float16)  # FFN up
    W_dn  = torch.randn(8192, 2048, device=dev, dtype=torch.float16)  # FFN down
    x     = torch.randn(1, 2048,   device=dev, dtype=torch.float16)   # hidden state
    res   = torch.randn(1, 2048,   device=dev, dtype=torch.float16)   # residual
    K_buf = torch.randn(32, 64, KV_LEN, device=dev, dtype=torch.float16)  # KV cache K
    V_buf = torch.randn(32, KV_LEN, 64, device=dev, dtype=torch.float16)  # KV cache V
    Q_h   = torch.randn(32, 1, 64,  device=dev, dtype=torch.float16)  # query heads

    def run_layers(n_layers):
        """Simulate n_layers of OPT-1.3b decode ops. Returns wall-clock ms."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_layers):
            # ① Linear projections (Q, K, V, O) — 4 GEMMs
            h = torch.mm(x, W_sq)
            h = torch.mm(x, W_sq)
            h = torch.mm(x, W_sq)
            h = torch.mm(h, W_sq)
            # ② LayerNorm (pre-attn)
            h = F.layer_norm(h, [2048])
            # ③ Residual add
            h = h + res
            # ④ Attn score: Q @ K^T  (32, 1, 64) × (32, 64, KV_LEN)
            score = torch.bmm(Q_h, K_buf)          # (32, 1, KV_LEN)
            # ⑤ Softmax
            score = F.softmax(score, dim=-1)
            # ⑥ Attn context: score @ V  (32, 1, KV_LEN) × (32, KV_LEN, 64)
            ctx = torch.bmm(score, V_buf)           # (32, 1, 64)
            # ⑦ FFN up
            h = torch.mm(x, W_up)
            # ⑧ FFN down
            h = torch.mm(h, W_dn)
            # ⑨ LayerNorm (pre-FFN)
            h = F.layer_norm(h, [2048])
            # ⑩ Residual add
            h = h + res
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3   # ms

    row = {"gpu_pct": gpu_pct, "layer_times": {}}

    for n in LAYER_COUNTS:
        # warm-up
        for _ in range(WARMUP_RUNS):
            run_layers(n)
        # measure
        times_ms = [run_layers(n) for _ in range(MEASURE_RUNS)]
        mean_ms  = float(np.mean(times_ms))
        std_ms   = float(np.std(times_ms))
        row["layer_times"][str(n)] = {"mean_ms": mean_ms, "std_ms": std_ms}
        print(f"[{gpu_pct}%]  {n:2d} layers: {mean_ms:7.3f} ± {std_ms:.3f} ms",
              file=sys.stderr, flush=True)

    print(json.dumps(row), flush=True)


# ─── Orchestrator ──────────────────────────────────────────────────────────
def orchestrate():
    all_results = []
    total = len(GPU_PERCENTAGES)
    for idx, pct in enumerate(GPU_PERCENTAGES, 1):
        print(f"\n{'─'*55}  [{idx}/{total}]  SM={pct}%")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]             = "0"
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)
        proc = subprocess.run(
            [sys.executable, __file__, "--mode=worker"],
            env=env, capture_output=True, text=True, timeout=600,
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


# ─── Plot & Analysis ────────────────────────────────────────────────────────
def plot(results=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if results is None:
        results = json.loads(RESULTS_FILE.read_text())

    by_pct = {r["gpu_pct"]: r for r in results}
    pcts   = sorted(by_pct.keys())
    ns     = sorted(int(k) for k in next(iter(by_pct.values()))["layer_times"])

    # colour map: spike candidates red, neighbours blue/green
    pct_style = {
        30:  ("#e74c3c", "o", "-",  "30% SM  ← spike"),
        40:  ("#2980b9", "s", "--", "40% SM"),
        70:  ("#c0392b", "^", "-",  "70% SM  ← spike"),
        80:  ("#27ae60", "D", "--", "80% SM"),
        100: ("#7f8c8d", "x", ":",  "100% SM (baseline)"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Spike Verification Round 3 — Layer-Count Scaling Test\n"
        "Hypothesis: extra latency at 30%/70% is linear in #layers (= accumulated kernel-launch overhead)",
        fontsize=12, fontweight="bold"
    )

    # ── Left: raw latency vs layer count ─────────────────────────────
    ax = axes[0]
    fits = {}
    for pct in pcts:
        r   = by_pct[pct]["layer_times"]
        xs  = np.array(ns, dtype=float)
        ys  = np.array([r[str(n)]["mean_ms"] for n in ns])
        es  = np.array([r[str(n)]["std_ms"]  for n in ns])
        col, mk, ls, lbl = pct_style[pct]

        ax.errorbar(xs, ys, yerr=es, label=lbl,
                    color=col, marker=mk, linestyle=ls,
                    linewidth=2, markersize=7, capsize=3)

        # linear fit: y = a*x + b
        a, b = np.polyfit(xs, ys, 1)
        fits[pct] = (a, b)
        ax.plot(xs, a*xs + b, color=col, linewidth=1, linestyle=":", alpha=0.6)

    ax.set_xlabel("Number of simulated decode layers", fontsize=11)
    ax.set_ylabel("Total wall-clock time (ms)", fontsize=11)
    ax.set_title("① Raw Latency vs Layer Count", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(ns)

    # ── Middle: slope (ms per layer) comparison ───────────────────────
    ax = axes[1]
    slope_pcts  = sorted(fits.keys())
    slope_vals  = [fits[p][0] for p in slope_pcts]
    bar_colors  = [pct_style[p][0] for p in slope_pcts]
    bars = ax.bar([str(p)+"%" for p in slope_pcts], slope_vals,
                  color=bar_colors, edgecolor="black", linewidth=0.8)
    # annotate values on bars
    for bar, v in zip(bars, slope_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("GPU SM Utilization (%)", fontsize=11)
    ax.set_ylabel("Slope: ms per layer (linear fit)", fontsize=11)
    ax.set_title("② Per-Layer Cost (slope)\n→ higher at 30%/70% if hypothesis holds",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # ── Right: predicted vs observed spike ────────────────────────────
    ax = axes[2]
    baseline_slope = fits[100][0]
    target_pcts    = [p for p in [30, 70] if p in fits]

    predicted_spike = {p: (fits[p][0] - baseline_slope) * 24 for p in target_pcts}
    # Observed spike from knee_results.json (hardcoded from experiment)
    observed_spike  = {30: 18.757 - 14.631, 70: 18.114 - 14.631}  # ms, input=512

    xs_labels = [f"{p}% SM" for p in target_pcts]
    x_pos     = np.arange(len(target_pcts))
    w         = 0.35

    bars_pred = ax.bar(x_pos - w/2,
                       [predicted_spike[p] for p in target_pcts],
                       w, label="Predicted (slope × 24 layers)",
                       color=["#e74c3c", "#c0392b"], edgecolor="black", linewidth=0.8)
    bars_obs  = ax.bar(x_pos + w/2,
                       [observed_spike.get(p, 0) for p in target_pcts],
                       w, label="Observed spike (full decode)",
                       color=["#f1948a", "#d98880"], edgecolor="black", linewidth=0.8,
                       hatch="//")

    for bar, v in zip(list(bars_pred) + list(bars_obs),
                      [predicted_spike[p] for p in target_pcts] +
                      [observed_spike.get(p, 0) for p in target_pcts]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.2f}ms", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(xs_labels)
    ax.set_ylabel("Extra latency vs 100% baseline (ms)", fontsize=11)
    ax.set_title("③ Predicted vs Observed Spike\n(if bars match → hypothesis confirmed)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {PLOT_FILE}")

    # ── Console summary ───────────────────────────────────────────────
    print("\n" + "═"*65)
    print("Linear fit results: time(ms) = slope × layers + intercept")
    print("─"*65)
    print(f"{'SM%':>5}  {'slope (ms/layer)':>18}  {'intercept':>12}  {'R²':>8}")
    print("─"*65)
    for pct in pcts:
        a, b = fits[pct]
        ys_hat = a * np.array(ns) + b
        ys_act = np.array([by_pct[pct]["layer_times"][str(n)]["mean_ms"] for n in ns])
        ss_res = np.sum((ys_act - ys_hat)**2)
        ss_tot = np.sum((ys_act - np.mean(ys_act))**2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        print(f"{pct:>4}%  {a:>18.4f}  {b:>12.4f}  {r2:>8.4f}")
    print("─"*65)
    print(f"\nPer-layer overhead at spike SM% vs 100% baseline:")
    for p in target_pcts:
        extra_per_layer = fits[p][0] - baseline_slope
        predicted_total = extra_per_layer * 24
        observed_total  = observed_spike.get(p, float("nan"))
        match = "✓ MATCH" if abs(predicted_total - observed_total) < 1.0 else "✗ MISMATCH"
        print(f"  {p}% SM: extra {extra_per_layer:.4f} ms/layer × 24 = "
              f"{predicted_total:.3f} ms  (observed: {observed_total:.3f} ms)  {match}")
    print("═"*65)


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

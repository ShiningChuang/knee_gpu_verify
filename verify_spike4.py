#!/usr/bin/env python3
"""
Spike Verification Round 4 — Non-kernel Time Decomposition
===========================================================

Goal: Distinguish where the "non-kernel time" at 70% SM goes.

  t_wall  = t_dispatch + t_flush
           where:
             t_dispatch = time for model() call before any sync
                          (CPU Python overhead + async kernel submission)
             t_flush    = time for torch.cuda.synchronize() after model()
                          (GPU still running after CPU finishes)

  Σkernels = sum of individual CUDA kernel durations (from profiler)

  inter_kernel_gap = t_flush - (Σkernels - already_overlapped)
    → approximated via Chrome trace: time from first kernel start
      to last kernel end, minus Σkernels = total GPU idle within the span

Hypothesis breakdown:
  If spike is CPU-busy        → t_dispatch(70%) >> t_dispatch(80%)
  If spike is GPU inter-gaps  → t_flush(70%)    >> t_flush(80%)
                                  AND chrome_gap(70%) >> chrome_gap(80%)
  If spike is sync/wait       → t_flush large, chrome_gap small
                                  (GPU runs solid but late)

Experiment A: dispatch/flush split across SM%
Experiment B: Chrome trace inter-kernel gap analysis
"""

import os, sys, json, time, argparse, subprocess, tempfile
from pathlib import Path
import numpy as np

GPU_PERCENTAGES = [40, 30, 80, 70]
INPUT_LEN       = 512
WARMUP_STEPS    = 5
MEASURE_STEPS   = 20      # for A (timing)
TRACE_STEPS     = 5       # for B (Chrome trace, fewer to keep file small)
MODEL_NAME      = "facebook/opt-1.3b"
RESULTS_FILE    = Path(__file__).parent / "spike4_results.json"
PLOT_FILE       = Path(__file__).parent / "spike4_result.png"


# ─── Worker ────────────────────────────────────────────────────────────────
def worker():
    import torch
    from torch.profiler import profile, ProfilerActivity
    from transformers import AutoModelForCausalLM

    gpu_pct = int(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", 100))
    assert torch.cuda.is_available()
    log = lambda m: print(f"[{gpu_pct}%] {m}", file=sys.stderr, flush=True)

    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16,
        device_map={"": 0}, low_cpu_mem_usage=True, use_safetensors=True,
    )
    model.eval()
    log("Model loaded.")

    # Prefill to build KV cache
    ids = torch.ones((1, INPUT_LEN), dtype=torch.long, device="cuda:0")
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past_kv  = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(-1)

    # ── Warm-up ─────────────────────────────────────────────────────────
    log(f"Warming up {WARMUP_STEPS} steps...")
    for _ in range(WARMUP_STEPS):
        torch.cuda.synchronize()
        with torch.no_grad():
            out = model(next_tok, past_key_values=past_kv, use_cache=True)
        torch.cuda.synchronize()
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(-1)

    # ═══════════════════════════════════════════════════════════════════
    # Experiment A: dispatch / flush decomposition
    # ═══════════════════════════════════════════════════════════════════
    dispatch_ms_list, flush_ms_list, wall_ms_list = [], [], []

    log(f"Experiment A: {MEASURE_STEPS} steps dispatch/flush split...")
    for _ in range(MEASURE_STEPS):
        torch.cuda.synchronize()          # ensure GPU is idle before each step

        # ── t_dispatch: CPU dispatch time (model() returns before GPU done) ──
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(next_tok, past_key_values=past_kv, use_cache=True)
        t1 = time.perf_counter()

        # ── t_flush: GPU residual run time after model() returned ────────────
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        dispatch_ms = (t1 - t0) * 1e3
        flush_ms    = (t2 - t1) * 1e3
        wall_ms     = (t2 - t0) * 1e3

        dispatch_ms_list.append(dispatch_ms)
        flush_ms_list.append(flush_ms)
        wall_ms_list.append(wall_ms)

        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(-1)

    dispatch_mean = float(np.mean(dispatch_ms_list))
    dispatch_std  = float(np.std(dispatch_ms_list))
    flush_mean    = float(np.mean(flush_ms_list))
    flush_std     = float(np.std(flush_ms_list))
    wall_mean     = float(np.mean(wall_ms_list))

    log(f"  t_dispatch = {dispatch_mean:.3f} ± {dispatch_std:.3f} ms")
    log(f"  t_flush    = {flush_mean:.3f} ± {flush_std:.3f} ms")
    log(f"  t_wall     = {wall_mean:.3f} ms")

    # ═══════════════════════════════════════════════════════════════════
    # Experiment B: Chrome trace → inter-kernel gap analysis
    # ═══════════════════════════════════════════════════════════════════
    trace_path = Path(tempfile.mktemp(suffix=".json"))
    log(f"Experiment B: Chrome trace ({TRACE_STEPS} steps) → {trace_path.name}")

    # Snapshot of KV state for trace (don't corrupt main state)
    trace_past_kv  = past_kv
    trace_next_tok = next_tok

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(TRACE_STEPS):
            with torch.no_grad():
                out = model(trace_next_tok,
                            past_key_values=trace_past_kv, use_cache=True)
            torch.cuda.synchronize()
            trace_past_kv  = out.past_key_values
            trace_next_tok = out.logits[:, -1:, :].argmax(-1)

    prof.export_chrome_trace(str(trace_path))
    log(f"  Trace exported ({trace_path.stat().st_size/1024:.0f} KB). Parsing gaps...")

    # ── Parse Chrome trace to find inter-kernel gaps ──────────────────
    with open(trace_path) as f:
        trace_data = json.load(f)

    events = trace_data if isinstance(trace_data, list) else trace_data.get("traceEvents", [])

    # Keep only CUDA kernel events (ph="X" = complete event, on GPU process)
    kernel_events = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        # CUDA kernel events have a "cat" field containing "kernel"
        cat = ev.get("cat", "").lower()
        if "kernel" not in cat and "cuda" not in cat:
            continue
        ts  = ev.get("ts", 0)    # start µs
        dur = ev.get("dur", 0)   # duration µs
        if dur > 0:
            kernel_events.append((ts, ts + dur, dur, ev.get("name", "")))

    kernel_events.sort(key=lambda x: x[0])  # sort by start time

    # Compute inter-kernel gaps (gap between end of kernel i and start of kernel i+1)
    gaps_us = []
    kernel_durations_us = []
    for i, (start, end, dur, name) in enumerate(kernel_events):
        kernel_durations_us.append(dur)
        if i > 0:
            prev_end = kernel_events[i-1][1]
            gap      = start - prev_end
            if gap > 0:    # only positive gaps (negative = overlap, shouldn't happen on 1 GPU)
                gaps_us.append(gap)

    total_kernel_us  = sum(kernel_durations_us)
    total_gap_us     = sum(gaps_us)
    n_kernels        = len(kernel_events)
    n_gaps           = len(gaps_us)
    mean_gap_us      = float(np.mean(gaps_us)) if gaps_us else 0.0
    p95_gap_us       = float(np.percentile(gaps_us, 95)) if gaps_us else 0.0
    span_us          = (kernel_events[-1][1] - kernel_events[0][0]) if kernel_events else 0
    gpu_busy_pct     = 100.0 * total_kernel_us / span_us if span_us > 0 else 0.0

    # Per-step averages
    per_step = lambda x: x / TRACE_STEPS
    gap_per_step_us    = per_step(total_gap_us)
    kernel_per_step_us = per_step(total_kernel_us)

    log(f"  Kernels: {n_kernels} ({n_kernels/TRACE_STEPS:.0f}/step)  "
        f"Gaps: {n_gaps} ({n_gaps/TRACE_STEPS:.0f}/step)")
    log(f"  GPU kernel time /step: {kernel_per_step_us/1e3:.3f} ms")
    log(f"  Inter-kernel gap/step: {gap_per_step_us/1e3:.3f} ms  "
        f"(mean gap: {mean_gap_us:.1f} µs, p95: {p95_gap_us:.1f} µs)")
    log(f"  GPU busy fraction: {gpu_busy_pct:.1f}%")
    trace_path.unlink(missing_ok=True)

    # ── Output result ──────────────────────────────────────────────────
    result = {
        "gpu_pct"           : gpu_pct,
        "input_len"         : INPUT_LEN,
        # Experiment A
        "dispatch_ms"       : dispatch_mean,
        "dispatch_std"      : dispatch_std,
        "flush_ms"          : flush_mean,
        "flush_std"         : flush_std,
        "wall_ms"           : wall_mean,
        # Experiment B
        "kernel_sum_ms"     : kernel_per_step_us / 1e3,
        "gap_sum_ms"        : gap_per_step_us / 1e3,
        "mean_gap_us"       : mean_gap_us,
        "p95_gap_us"        : p95_gap_us,
        "n_kernels_per_step": n_kernels / TRACE_STEPS,
        "gpu_busy_pct"      : gpu_busy_pct,
    }
    print(json.dumps(result), flush=True)


# ─── Orchestrator ──────────────────────────────────────────────────────────
def orchestrate():
    all_results = []
    for idx, pct in enumerate(GPU_PERCENTAGES, 1):
        print(f"\n{'═'*60}  [{idx}/{len(GPU_PERCENTAGES)}]  SM={pct}%")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]             = "0"
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)
        proc = subprocess.run(
            [sys.executable, __file__, "--mode=worker"],
            env=env, capture_output=True, text=True, timeout=900,
        )
        for line in proc.stderr.strip().splitlines():
            print(f"  {line}")
        for line in proc.stdout.strip().splitlines():
            if line.startswith("{"):
                all_results.append(json.loads(line))
                break
        else:
            print(f"  [WARN] no result  rc={proc.returncode}")
            print(proc.stderr[-400:])
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\n[Done] {len(all_results)} results → {RESULTS_FILE}")
    return all_results


# ─── Plot ──────────────────────────────────────────────────────────────────
def plot(results=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if results is None:
        results = json.loads(RESULTS_FILE.read_text())

    by_pct = {r["gpu_pct"]: r for r in results}
    pcts   = sorted(by_pct.keys())
    xlbls  = [f"{p}%\n{'▲spike' if p in [30,70] else ''}" for p in pcts]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Spike Verification Round 4 — Non-kernel Time Decomposition\n"
        "Exp A: dispatch/flush split  |  Exp B: Chrome trace inter-kernel gaps",
        fontsize=12, fontweight="bold"
    )

    spike_clr  = "#e74c3c"
    normal_clr = "#2980b9"
    bar_colors = [spike_clr if p in [30, 70] else normal_clr for p in pcts]

    # ── Plot 1: Stacked bar — dispatch / flush / (labelled) ───────────
    ax = axes[0]
    dispatch_vals = [by_pct[p]["dispatch_ms"] for p in pcts]
    flush_vals    = [by_pct[p]["flush_ms"]    for p in pcts]
    x = np.arange(len(pcts))
    b1 = ax.bar(x, dispatch_vals, label="t_dispatch  (CPU: Python + kernel submit)",
                color="#3498db", edgecolor="white")
    b2 = ax.bar(x, flush_vals, bottom=dispatch_vals,
                label="t_flush  (GPU residual after model() returns)",
                color="#e67e22", edgecolor="white")
    # total label
    for i, p in enumerate(pcts):
        total = dispatch_vals[i] + flush_vals[i]
        ax.text(i, total + 0.1, f"{total:.2f}ms", ha="center", fontsize=9, fontweight="bold")
    # annotate each segment
    for i, (dv, fv) in enumerate(zip(dispatch_vals, flush_vals)):
        if dv > 0.5: ax.text(i, dv/2, f"{dv:.2f}", ha="center", va="center", color="white", fontsize=8)
        if fv > 0.3: ax.text(i, dv + fv/2, f"{fv:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=10)
    ax.set_ylabel("Time per decode step (ms)", fontsize=11)
    ax.set_title("① t_dispatch vs t_flush\n(which component carries the spike?)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # ── Plot 2: inter-kernel gap per step ─────────────────────────────
    ax = axes[1]
    gap_vals  = [by_pct[p]["gap_sum_ms"]     for p in pcts]
    kern_vals = [by_pct[p]["kernel_sum_ms"]  for p in pcts]
    ax.bar(x - 0.2, kern_vals, width=0.35, label="GPU kernel execution",
           color="#27ae60", edgecolor="white")
    ax.bar(x + 0.2, gap_vals,  width=0.35, label="Inter-kernel GPU gaps",
           color=bar_colors, edgecolor="white")
    for i, (g, p) in enumerate(zip(gap_vals, pcts)):
        ax.text(i + 0.2, g + 0.05, f"{g:.3f}ms", ha="center", fontsize=8,
                color=spike_clr if p in [30,70] else "black", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=10)
    ax.set_ylabel("Time per decode step (ms)", fontsize=11)
    ax.set_title("② GPU kernel time vs inter-kernel gaps\n(from Chrome trace)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # ── Plot 3: mean gap per kernel launch ────────────────────────────
    ax = axes[2]
    mean_gaps = [by_pct[p]["mean_gap_us"]    for p in pcts]
    p95_gaps  = [by_pct[p]["p95_gap_us"]     for p in pcts]
    n_kerns   = [by_pct[p]["n_kernels_per_step"] for p in pcts]
    ax.bar(x - 0.2, mean_gaps, width=0.35, label="Mean gap (µs)",
           color=bar_colors, edgecolor="white")
    ax.bar(x + 0.2, p95_gaps,  width=0.35, label="P95 gap (µs)",
           color=[c + "88" for c in bar_colors], edgecolor="white", hatch="//")
    for i, (mg, p) in enumerate(zip(mean_gaps, pcts)):
        ax.text(i - 0.2, mg + 0.3, f"{mg:.1f}", ha="center", fontsize=8)
    for i, nk in enumerate(n_kerns):
        ax.text(i, -4, f"n={nk:.0f}", ha="center", fontsize=7, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=10)
    ax.set_ylabel("Inter-kernel gap (µs)", fontsize=11)
    ax.set_title("③ Per-gap statistics\n(larger mean/p95 at spike SM% → MPS queuing)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {PLOT_FILE}")

    # ── Console summary ───────────────────────────────────────────────
    print("\n" + "═"*80)
    hdr = f"{'SM%':>4}  {'t_dispatch':>12} {'t_flush':>12} {'t_wall':>9}"
    hdr += f"  {'kern_sum':>9} {'gap_sum':>9} {'mean_gap':>10} {'busy%':>7}"
    print(hdr)
    print("─"*80)
    for p in pcts:
        r = by_pct[p]
        flag = " ←" if p in [30, 70] else "  "
        print(f"{p:>3}%{flag}  "
              f"{r['dispatch_ms']:>8.3f}ms  {r['flush_ms']:>8.3f}ms  "
              f"{r['wall_ms']:>7.3f}ms  "
              f"{r['kernel_sum_ms']:>7.3f}ms  {r['gap_sum_ms']:>7.3f}ms  "
              f"{r['mean_gap_us']:>8.1f}µs  {r['gpu_busy_pct']:>6.1f}%")
    print("═"*80)
    print("\nDiagnostic interpretation:")
    ref = by_pct.get(80, by_pct[max(by_pct)])
    for spike_p in [30, 70]:
        if spike_p not in by_pct:
            continue
        s = by_pct[spike_p]
        dd = s["dispatch_ms"] - ref["dispatch_ms"]
        df = s["flush_ms"]    - ref["flush_ms"]
        dg = s["gap_sum_ms"]  - ref["gap_sum_ms"]
        print(f"\n  {spike_p}% vs 80% baseline:")
        print(f"    Δt_dispatch = {dd:+.3f} ms  "
              f"{'← CPU is busier at {spike_p}%' if dd > 0.5 else '(CPU overhead unchanged)'}")
        print(f"    Δt_flush    = {df:+.3f} ms  "
              f"{'← GPU runs longer after model() returns' if df > 0.5 else '(flush unchanged)'}")
        print(f"    Δgap_sum    = {dg:+.3f} ms  "
              f"{'← more inter-kernel GPU idle time' if dg > 0.3 else '(gaps unchanged)'}")
        dom = max([("CPU dispatch", abs(dd)), ("GPU flush", abs(df)),
                   ("inter-kernel gap", abs(dg))], key=lambda x: x[1])
        print(f"    → Dominant contributor: {dom[0]} ({dom[1]:.3f} ms extra)")


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

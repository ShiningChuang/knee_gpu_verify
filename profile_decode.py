#!/usr/bin/env python3
"""
CUDA Kernel-Level Profiling — OPT-1.3b Decode at Different SM%
===============================================================

Uses torch.profiler (CUPTI) to capture per-kernel CUDA execution times
inside the actual OPT-1.3b model forward pass at each MPS SM%.

Goal: pinpoint which CUDA kernel type is responsible for the decode
latency spikes at 30% and 70% SM.

Kernel classification (by name prefix):
  GEMM/tensor-core : ampere_h16816gemm / gemv2N_kernel / splitKreduce /
                     volta_h884gemm / cublasGemm*
  LayerNorm        : vectorized_layer_norm / layer_norm*
  Softmax          : softmax* / _softmax*
  Elementwise      : elementwise_kernel / vectorized_elementwise*
  Memory/Other     : everything else (cat, index, copy, etc.)
"""

import os, sys, json, argparse, subprocess, time
from pathlib import Path
import numpy as np

GPU_PERCENTAGES = [40, 30, 80, 70]    # include neighbours for comparison
INPUT_LEN       = 512
DECODE_STEPS    = 10     # profile this many decode steps (averaged)
WARMUP_STEPS    = 3      # steps to skip before profiling
RESULTS_FILE    = Path(__file__).parent / "profile_results.json"
PLOT_FILE       = Path(__file__).parent / "profile_result.png"
MODEL_NAME      = "facebook/opt-1.3b"


# ── kernel name → category mapping ─────────────────────────────────────────
def classify_kernel(name: str) -> str:
    n = name.lower()
    # GEMM / tensor-core matrix multiply
    if any(k in n for k in ["gemm", "gemv", "splitkreduce", "volta_h", "ampere_h",
                             "cutlass", "wgrad", "dgrad", "fprop"]):
        return "GEMM / MatMul"
    # Layer norm
    if any(k in n for k in ["layer_norm", "vectorized_layer", "rms_norm"]):
        return "LayerNorm"
    # Softmax / attention mask
    if any(k in n for k in ["softmax", "_softmax"]):
        return "Softmax"
    # Elementwise (add, mul, gelu, relu, etc.)
    if any(k in n for k in ["elementwise", "vectorized_element", "unary", "binary",
                             "add_", "fused_dropout", "gelu", "relu", "silu"]):
        return "Elementwise"
    # Memory ops (cat, index, copy, gather, scatter)
    if any(k in n for k in ["cat", "index", "copy", "gather", "scatter",
                             "memcpy", "memset", "fill", "clone", "contiguous",
                             "concat", "select", "view", "reshape", "transpose",
                             "embedding", "lut"]):
        return "Memory / Indexing"
    return "Other"


# ─── Worker ────────────────────────────────────────────────────────────────
def worker():
    import torch
    import torch.nn.functional as F
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

    # Build initial input and KV cache via a prefill pass
    input_ids = torch.ones((1, INPUT_LEN), dtype=torch.long, device="cuda:0")
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past_kv  = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(-1)

    # Warm up decode (not profiled)
    log(f"Warming up {WARMUP_STEPS} decode steps...")
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            out = model(next_tok, past_key_values=past_kv, use_cache=True)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(-1)

    # Profile DECODE_STEPS steps
    log(f"Profiling {DECODE_STEPS} decode steps...")
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for step in range(DECODE_STEPS):
            with torch.no_grad():
                out = model(next_tok, past_key_values=past_kv, use_cache=True)
            past_kv  = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(-1)
            torch.cuda.synchronize()

    # Aggregate per-kernel stats
    kernel_stats = {}   # name → {cuda_us, count}
    for evt in prof.key_averages():
        if evt.self_device_time_total <= 0:
            continue
        name = evt.key
        if name not in kernel_stats:
            kernel_stats[name] = {"cuda_us": 0.0, "count": 0}
        kernel_stats[name]["cuda_us"] += evt.self_device_time_total
        kernel_stats[name]["count"]   += evt.count

    # Classify into categories
    category_us = {}
    kernel_detail = []
    for name, st in kernel_stats.items():
        cat  = classify_kernel(name)
        us   = st["cuda_us"]
        cnt  = st["count"]
        category_us[cat] = category_us.get(cat, 0.0) + us
        kernel_detail.append({
            "name": name, "category": cat,
            "total_cuda_us": us, "count": cnt,
            "avg_us": us / cnt if cnt > 0 else 0,
        })

    # Per-step averages
    cat_per_step = {cat: us / DECODE_STEPS
                    for cat, us in category_us.items()}
    total_per_step = sum(cat_per_step.values())

    log(f"Total CUDA time per decode step: {total_per_step/1e3:.3f} ms")
    for cat, us in sorted(cat_per_step.items(), key=lambda x: -x[1]):
        log(f"  {cat:25s}: {us/1e3:7.3f} ms  ({100*us/total_per_step:.1f}%)")

    # Top-10 kernels
    top10 = sorted(kernel_detail, key=lambda x: -x["total_cuda_us"])[:10]

    result = {
        "gpu_pct":       gpu_pct,
        "decode_steps":  DECODE_STEPS,
        "input_len":     INPUT_LEN,
        "categories":    cat_per_step,           # µs per decode step per category
        "total_us":      total_per_step,
        "top_kernels":   top10,
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
    import matplotlib.ticker as ticker

    if results is None:
        results = json.loads(RESULTS_FILE.read_text())

    by_pct = {r["gpu_pct"]: r for r in results}
    pcts   = sorted(by_pct.keys())

    # Collect all categories
    all_cats = set()
    for r in results:
        all_cats.update(r["categories"].keys())
    # Fixed display order
    cat_order = ["GEMM / MatMul", "LayerNorm", "Softmax",
                 "Elementwise", "Memory / Indexing", "Other"]
    cats = [c for c in cat_order if c in all_cats] + \
           [c for c in all_cats  if c not in cat_order]

    cat_colors = {
        "GEMM / MatMul"    : "#2196F3",
        "LayerNorm"        : "#FF9800",
        "Softmax"          : "#4CAF50",
        "Elementwise"      : "#9C27B0",
        "Memory / Indexing": "#F44336",
        "Other"            : "#9E9E9E",
    }

    pct_labels = {p: f"{p}% SM" + (" ←spike" if p in [30, 70] else "") for p in pcts}

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"CUDA Kernel Profiling — OPT-1.3b Decode, input_len={INPUT_LEN}, "
        f"{DECODE_STEPS} steps (torch.profiler + CUPTI)\n"
        "Comparing spike SM% (30%, 70%) vs neighbours (40%, 80%)",
        fontsize=13, fontweight="bold"
    )

    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4)

    # ── Plot 1: Stacked bar — time per category per SM% ──────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x   = np.arange(len(pcts))
    bottoms = np.zeros(len(pcts))
    for cat in cats:
        vals = [by_pct[p]["categories"].get(cat, 0) / 1e3 for p in pcts]  # → ms
        bars = ax1.bar(x, vals, bottom=bottoms, label=cat,
                       color=cat_colors.get(cat, "#9E9E9E"),
                       edgecolor="white", linewidth=0.5)
        # annotate significant categories
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v > 0.3:   # only annotate if > 0.3ms
                ax1.text(bar.get_x() + bar.get_width()/2,
                         bottoms[i] + v/2,
                         f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color="white", fontweight="bold")
        bottoms += np.array(vals)

    # total labels on top
    for i, p in enumerate(pcts):
        total = by_pct[p]["total_us"] / 1e3
        ax1.text(i, bottoms[i] + 0.1, f"{total:.2f} ms",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([pct_labels[p] for p in pcts], fontsize=10)
    ax1.set_ylabel("CUDA time per decode step (ms)", fontsize=11)
    ax1.set_title("① Total CUDA time breakdown by kernel category", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--", axis="y")

    # ── Plot 2: Per-category comparison across SM% ───────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    for cat in cats:
        ys = [by_pct[p]["categories"].get(cat, 0) / 1e3 for p in pcts]
        ax2.plot(pcts, ys, marker="o", label=cat,
                 color=cat_colors.get(cat, "#9E9E9E"),
                 linewidth=2, markersize=7)
    ax2.set_xlabel("GPU SM Utilization (%)", fontsize=10)
    ax2.set_ylabel("CUDA time per step (ms)", fontsize=10)
    ax2.set_title("② Per-category vs SM%\n(look for lines rising at 30%/70%)",
                  fontsize=10, fontweight="bold")
    ax2.set_xticks(pcts)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3, linestyle="--")
    for p in [30, 70]:
        if p in pcts:
            ax2.axvline(p, color="red", linestyle=":", linewidth=1.2, alpha=0.7)

    # ── Plot 3-6: Top-10 kernels per SM% ─────────────────────────────
    for subplot_idx, pct in enumerate(pcts):
        r     = by_pct[pct]
        ax    = fig.add_subplot(gs[1, subplot_idx % 3]) if subplot_idx < 3 else None
        if ax is None:
            continue
        top10 = sorted(r["top_kernels"], key=lambda x: -x["total_cuda_us"])[:8]
        names = [t["name"][:35] + "…" if len(t["name"]) > 35 else t["name"]
                 for t in top10]
        vals  = [t["total_cuda_us"] / (DECODE_STEPS * 1e3) for t in top10]  # ms/step
        colors_bar = [cat_colors.get(t["category"], "#9E9E9E") for t in top10]

        bars = ax.barh(range(len(names)), vals[::-1], color=colors_bar[::-1],
                       edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1], fontsize=6.5)
        ax.set_xlabel("ms per decode step", fontsize=9)
        spike_tag = " ← SPIKE" if pct in [30, 70] else ""
        ax.set_title(f"③ Top kernels — {pct}% SM{spike_tag}",
                     fontsize=10, fontweight="bold",
                     color="red" if pct in [30, 70] else "black")
        ax.grid(True, alpha=0.3, linestyle="--", axis="x")

    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved → {PLOT_FILE}")

    # ── Console report ────────────────────────────────────────────────
    print("\n" + "═"*85)
    print(f"{'Category':25s}", end="")
    for p in pcts:
        print(f"  {p}%SM(ms)", end="")
    print()
    print("─"*85)
    for cat in cats:
        print(f"{cat:25s}", end="")
        baseline = by_pct[pcts[-1]]["categories"].get(cat, 0) / 1e3
        for p in pcts:
            ms = by_pct[p]["categories"].get(cat, 0) / 1e3
            flag = " ★" if (p in [30,70] and baseline>0 and ms/baseline>1.2) else "  "
            print(f"  {ms:6.3f}{flag}", end="")
        print()
    print("─"*85)
    print(f"{'TOTAL':25s}", end="")
    for p in pcts:
        print(f"  {by_pct[p]['total_us']/1e3:6.3f}  ", end="")
    print()
    print("═"*85)
    print("★ = >20% slower than rightmost SM% baseline")

    # Top kernels diff table (spike vs neighbour)
    print("\n── Top kernels unique to or slower at SPIKE SM% ──")
    for spike_pct, ref_pct in [(30, 40), (70, 80)]:
        if spike_pct not in by_pct or ref_pct not in by_pct:
            continue
        print(f"\n  Comparing {spike_pct}% (spike) vs {ref_pct}% (neighbour):")
        spike_kernels = {k["name"]: k["total_cuda_us"] / DECODE_STEPS
                         for k in by_pct[spike_pct]["top_kernels"]}
        ref_kernels   = {k["name"]: k["total_cuda_us"] / DECODE_STEPS
                         for k in by_pct[ref_pct]["top_kernels"]}
        all_k = set(spike_kernels) | set(ref_kernels)
        diffs = []
        for name in all_k:
            s = spike_kernels.get(name, 0)
            r = ref_kernels.get(name, 0)
            diffs.append((name, s, r, s - r))
        for name, s, r, diff in sorted(diffs, key=lambda x: -abs(x[3]))[:8]:
            cat = classify_kernel(name)
            direction = "↑SLOWER" if diff > 0 else "↓faster"
            print(f"    [{cat:20s}] {diff/1e3:+7.3f}ms/step  {direction}"
                  f"  {name[:50]}")


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

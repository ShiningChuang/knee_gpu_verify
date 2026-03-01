#!/usr/bin/env python3
"""
Multi-Model Knee GPU% Comparison
=================================

Tests the "knee GPU%" phenomenon across four models representing
different scales and architectures common in current research:

  Model              Size   Arch       Key feature
  ─────────────────────────────────────────────────────────────────
  facebook/opt-1.3b  1.3B   OPT        baseline (already measured)
  Qwen/Qwen2-1.5B    1.5B   Qwen2      GQA, RoPE — modern design
  microsoft/phi-2    2.7B   Phi        SwiGLU, dense MLP, efficient
  facebook/opt-6.7b  6.7B   OPT        same family as 1.3b, 5× larger

Hypothesis to verify:
  • Prefill knee shifts RIGHT (higher SM%) as model grows
    (larger matmuls saturate more SMs before plateauing)
  • Decode knee stays LEFT (low SM%) for all models
    (decode is memory-BW bound regardless of architecture)
  • Architecture affects knee location less than model size

Usage:
  python knee_multimodel.py                # full experiment
  python knee_multimodel.py --mode plot    # re-plot from JSON
"""

import os, sys, json, time, argparse, subprocess
from pathlib import Path
import numpy as np

# ─── Configuration ─────────────────────────────────────────────────────────
MODELS = [
    {"id": "facebook/opt-1.3b",  "name": "OPT-1.3B",  "params_b": 1.3, "color": "#1f77b4"},
    {"id": "Qwen/Qwen2-1.5B",    "name": "Qwen2-1.5B", "params_b": 1.5, "color": "#ff7f0e"},
    {"id": "microsoft/phi-2",    "name": "Phi-2",       "params_b": 2.7, "color": "#2ca02c"},
    {"id": "facebook/opt-6.7b",  "name": "OPT-6.7B",   "params_b": 6.7, "color": "#d62728"},
]
GPU_PERCENTAGES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
INPUT_LEN       = 512      # fixed for clean cross-model comparison
DECODE_STEPS    = 30
WARMUP_RUNS     = 2
MEASURE_RUNS    = 4
RESULTS_FILE    = Path(__file__).parent / "multimodel_results.json"
PLOT_FILE       = Path(__file__).parent / "multimodel_result.png"


# ─── Worker ────────────────────────────────────────────────────────────────
def worker(args):
    import torch
    from transformers import AutoModelForCausalLM

    model_id = args.model_id
    gpu_pct  = int(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", 100))
    assert torch.cuda.is_available()
    log = lambda m: print(f"[{gpu_pct}% {model_id.split('/')[-1]}] {m}",
                          file=sys.stderr, flush=True)

    log("Loading...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16,
            device_map={"": 0}, low_cpu_mem_usage=True, use_safetensors=True,
        )
    except Exception:
        # fallback: some models don't have safetensors
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16,
            device_map={"": 0}, low_cpu_mem_usage=True,
        )
    model.eval()
    mem_gb = torch.cuda.memory_allocated(0) / 1e9
    log(f"Loaded ({mem_gb:.2f} GB GPU)")

    # Prefill
    ids = torch.ones((1, INPUT_LEN), dtype=torch.long, device="cuda:0")
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past_kv  = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(-1)

    def measure_once():
        torch.cuda.synchronize()
        # Prefill
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1e3

        pk = out.past_key_values
        nt = out.logits[:, -1:, :].argmax(-1)

        # Decode
        steps = []
        for _ in range(DECODE_STEPS):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                out = model(nt, past_key_values=pk, use_cache=True)
            torch.cuda.synchronize()
            steps.append((time.perf_counter() - t1) * 1e3)
            pk = out.past_key_values
            nt = out.logits[:, -1:, :].argmax(-1)

        return prefill_ms, float(np.mean(steps))

    # Warm-up
    for _ in range(WARMUP_RUNS):
        measure_once()

    # Measure
    pf, dc = [], []
    for _ in range(MEASURE_RUNS):
        p, d = measure_once()
        pf.append(p); dc.append(d)

    result = {
        "model_id":   model_id,
        "gpu_pct":    gpu_pct,
        "input_len":  INPUT_LEN,
        "prefill_ms": float(np.mean(pf)),
        "prefill_std":float(np.std(pf)),
        "decode_ms":  float(np.mean(dc)),
        "decode_std": float(np.std(dc)),
        "mem_gb":     mem_gb,
    }
    log(f"prefill={result['prefill_ms']:.2f}ms  decode={result['decode_ms']:.3f}ms/tok")
    print(json.dumps(result), flush=True)


# ─── Orchestrator ──────────────────────────────────────────────────────────
def orchestrate():
    all_results = []
    configs = [(m, p) for m in MODELS for p in GPU_PERCENTAGES]
    total   = len(configs)

    for idx, (model_cfg, pct) in enumerate(configs, 1):
        mid  = model_cfg["id"]
        name = model_cfg["name"]
        print(f"\n{'═'*60}")
        print(f"  [{idx:3d}/{total}]  {name} ({model_cfg['params_b']}B)  SM={pct}%")
        print(f"{'═'*60}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]             = "0"
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)

        proc = subprocess.run(
            [sys.executable, __file__, "--mode=worker", f"--model_id={mid}"],
            env=env, capture_output=True, text=True, timeout=1200,
        )
        for line in proc.stderr.strip().splitlines():
            print(f"  {line}")

        for line in proc.stdout.strip().splitlines():
            if line.startswith("{"):
                all_results.append(json.loads(line))
                break
        else:
            print(f"  [WARN] no result  rc={proc.returncode}")
            if proc.returncode != 0:
                print(proc.stderr[-300:])

        # Checkpoint after each SM config
        RESULTS_FILE.write_text(json.dumps(all_results, indent=2))

    print(f"\n[Done] {len(all_results)} results → {RESULTS_FILE}")
    return all_results


# ─── Plot ──────────────────────────────────────────────────────────────────
def plot(results=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from collections import defaultdict

    if results is None:
        results = json.loads(RESULTS_FILE.read_text())
    if not results:
        print("[ERROR] No results to plot."); return

    # Organise: data[model_id][gpu_pct] = row
    data   = defaultdict(dict)
    for r in results:
        data[r["model_id"]][r["gpu_pct"]] = r

    pcts      = sorted({r["gpu_pct"] for r in results})
    model_ids = [m["id"] for m in MODELS if m["id"] in data]
    model_map = {m["id"]: m for m in MODELS}

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Knee GPU% — Multi-Model Comparison\n"
        "OPT-1.3B | Qwen2-1.5B | Phi-2 | OPT-6.7B  ·  Single V100, CUDA MPS",
        fontsize=14, fontweight="bold"
    )

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    # ── Row 0: Prefill & Decode raw latency vs SM% ─────────────────
    for col, phase in enumerate(["prefill", "decode"]):
        ax = fig.add_subplot(gs[0, col])
        key_ms  = f"{phase}_ms"
        key_std = f"{phase}_std"
        for mid in model_ids:
            mc  = model_map[mid]
            xs  = sorted(data[mid].keys())
            ys  = [data[mid][p][key_ms]  for p in xs]
            es  = [data[mid][p][key_std] for p in xs]
            ax.errorbar(xs, ys, yerr=es, label=f"{mc['name']} ({mc['params_b']}B)",
                        color=mc["color"], marker="o", linewidth=2, markersize=6,
                        capsize=3, capthick=1.2)
        ylabel = ("Prefill latency (ms)" if phase == "prefill"
                  else "Decode latency (ms / token)")
        title  = ("① Prefill Latency vs GPU%\n(compute-bound — knee shifts right for larger models)"
                  if phase == "prefill"
                  else "② Decode Latency vs GPU%\n(memory-BW bound — knee stays early for all)")
        ax.set_xlabel("GPU SM Utilization (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(pcts)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

    # ── Row 1: Normalised latency (relative to 100% SM baseline) ───
    for col, phase in enumerate(["prefill", "decode"]):
        ax = fig.add_subplot(gs[1, col])
        key_ms = f"{phase}_ms"
        for mid in model_ids:
            mc      = model_map[mid]
            xs      = sorted(data[mid].keys())
            base    = data[mid].get(100, {}).get(key_ms, None)
            if base is None or base == 0:
                continue
            ys = [data[mid][p][key_ms] / base for p in xs]
            ax.plot(xs, ys, label=f"{mc['name']}", color=mc["color"],
                    marker="o", linewidth=2, markersize=6)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("GPU SM Utilization (%)", fontsize=11)
        ax.set_ylabel("Normalised latency  (1.0 = 100% SM)", fontsize=11)
        ax.set_title(f"③ Normalised {phase.capitalize()} — Knee comparison\n"
                     "(curve flattens at knee → extra SMs give no benefit)",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(pcts)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

    # ── Row 2: Knee location estimation (where slope flattens) ─────
    ax_knee = fig.add_subplot(gs[2, :])

    # Find "knee" = smallest SM% where normalised latency < 1.15
    # (within 15% of 100% SM baseline)
    knee_data = {"prefill": {}, "decode": {}}
    for mid in model_ids:
        mc = model_map[mid]
        for phase in ["prefill", "decode"]:
            key_ms = f"{phase}_ms"
            base   = data[mid].get(100, {}).get(key_ms)
            if base is None: continue
            knee_pct = 100   # default: no knee found below 100%
            for p in sorted(data[mid].keys()):
                if data[mid][p][key_ms] / base < 1.15:
                    knee_pct = p
                    break
            knee_data[phase][mid] = knee_pct

    x      = np.arange(len(model_ids))
    w      = 0.35
    colors = [model_map[mid]["color"] for mid in model_ids]
    labels = [f"{model_map[mid]['name']}\n({model_map[mid]['params_b']}B)" for mid in model_ids]

    b1 = ax_knee.bar(x - w/2, [knee_data["prefill"].get(m, 100) for m in model_ids],
                     w, label="Prefill knee", color=colors,
                     edgecolor="black", linewidth=0.8)
    b2 = ax_knee.bar(x + w/2, [knee_data["decode"].get(m, 100) for m in model_ids],
                     w, label="Decode knee", color=colors, hatch="//",
                     edgecolor="black", linewidth=0.8, alpha=0.7)
    for bar, v in [(b, knee_data["prefill"].get(m, 100))
                   for b, m in zip(b1, model_ids)]:
        ax_knee.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                     f"{v}%", ha="center", fontsize=9, fontweight="bold")
    for bar, v in [(b, knee_data["decode"].get(m, 100))
                   for b, m in zip(b2, model_ids)]:
        ax_knee.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                     f"{v}%", ha="center", fontsize=9)

    ax_knee.set_xticks(x)
    ax_knee.set_xticklabels(labels, fontsize=10)
    ax_knee.set_ylabel("Knee SM% (first point within 15% of 100%-SM baseline)", fontsize=11)
    ax_knee.set_title(
        "④ Estimated Knee Location per Model\n"
        "(solid = prefill knee, hatched = decode knee  ·  lower = earlier knee = less benefit from more SMs)",
        fontsize=11, fontweight="bold"
    )
    ax_knee.legend(fontsize=10)
    ax_knee.set_ylim(0, 115)
    ax_knee.axhline(100, color="red", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_knee.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {PLOT_FILE}")

    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "═"*75)
    print(f"{'Model':14s}  {'Params':>7}  "
          f"{'Prefill@20%':>12}  {'Prefill@100%':>13}  "
          f"{'Decode@20%':>11}  {'Decode@100%':>12}  "
          f"{'PF-Knee':>8}  {'DC-Knee':>8}")
    print("─"*75)
    for mid in model_ids:
        mc  = model_map[mid]
        d   = data[mid]
        p20 = d.get(20, {})
        p100= d.get(100, {})
        print(f"{mc['name']:14s}  {mc['params_b']:>5.1f}B  "
              f"{p20.get('prefill_ms',0):>10.2f}ms  "
              f"{p100.get('prefill_ms',0):>11.2f}ms  "
              f"{p20.get('decode_ms',0):>9.3f}ms  "
              f"{p100.get('decode_ms',0):>10.3f}ms  "
              f"{knee_data['prefill'].get(mid,100):>7d}%  "
              f"{knee_data['decode'].get(mid,100):>7d}%")
    print("═"*75)


# ─── Entry ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-model knee GPU% comparison")
    parser.add_argument("--mode", default="orchestrate",
                        choices=["orchestrate", "worker", "plot"])
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    args = parser.parse_args()

    if args.mode == "orchestrate":
        results = orchestrate()
        plot(results)
    elif args.mode == "worker":
        worker(args)
    elif args.mode == "plot":
        plot()

if __name__ == "__main__":
    main()

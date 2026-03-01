#!/usr/bin/env python3
"""
Knee GPU% Verification — OPT-1.3b on Single V100 via CUDA MPS
==============================================================

方法：
  用 CUDA MPS 的 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 控制 SM 使用率
  (V100 共 80 个 SM，50% = 40 SM)

指标：
  - Prefill latency (ms)      : 处理输入序列、产出第一个 token 的时间
  - Decode latency (ms/token) : 每步自回归生成的平均时间

变量：
  - GPU SM% : 20, 30, 40, 50, 60, 70, 80, 90, 100
  - input_len: 128, 512, 1024 tokens

运行：
  # 完整实验
  python knee_verify.py

  # 只重绘图（有 knee_results.json 时）
  python knee_verify.py --mode plot
"""

import os, sys, json, time, argparse, subprocess, textwrap
from pathlib import Path
import numpy as np

# ──────────────────────────── 配置 ────────────────────────────────────────
MODEL_NAME      = "facebook/opt-1.3b"
GPU_PERCENTAGES = [20, 30, 40, 50, 60, 70, 80, 90, 100]   # SM 占比
INPUT_LENGTHS   = [128, 512, 1024]                          # prefill token 数
DECODE_STEPS    = 50     # 每次 decode 测多少步
WARMUP_RUNS     = 2      # 丢弃的热身次数
MEASURE_RUNS    = 5      # 正式测量次数
RESULTS_FILE    = Path(__file__).parent / "knee_results.json"
PLOT_FILE       = Path(__file__).parent / "knee_result.png"

# ── knee_test is the canonical location; paths auto-resolve via __file__ ──

# ──────────────────────────── Worker ──────────────────────────────────────
def worker(args):
    """
    子进程入口：继承父进程设置的 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE，
    在单卡上加载模型，对所有 input_len 进行测量，输出 JSON。
    """
    import torch
    from transformers import AutoModelForCausalLM

    gpu_pct = int(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", 100))
    assert torch.cuda.is_available(), "CUDA 不可用"

    log = lambda msg: print(f"[worker gpu%={gpu_pct}] {msg}", file=sys.stderr, flush=True)
    log(f"Loading {MODEL_NAME} in fp16 ...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map={"": 0},        # 强制整个模型在 cuda:0
        low_cpu_mem_usage=True,
        use_safetensors=True,      # 绕过 torch.load CVE-2025-32434 限制
    )
    model.eval()
    log("Model loaded.")

    def measure_latency(input_len):
        """对给定 input_len 测 prefill 和 decode 延迟，返回 (prefill_ms, decode_ms_per_tok)"""
        input_ids = torch.ones((1, input_len), dtype=torch.long, device="cuda:0")

        def single_pass():
            torch.cuda.synchronize()
            # ── Prefill ──────────────────────────────────────────────
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids, use_cache=True)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1e3

            # ── Decode ───────────────────────────────────────────────
            past_kv   = out.past_key_values
            next_tok  = out.logits[:, -1:, :].argmax(-1)   # (1,1)
            step_ms   = []
            for _ in range(DECODE_STEPS):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                with torch.no_grad():
                    out = model(next_tok, past_key_values=past_kv, use_cache=True)
                torch.cuda.synchronize()
                step_ms.append((time.perf_counter() - t1) * 1e3)
                past_kv  = out.past_key_values
                next_tok = out.logits[:, -1:, :].argmax(-1)

            return prefill_ms, float(np.mean(step_ms))

        # 热身
        for _ in range(WARMUP_RUNS):
            single_pass()

        # 正式测量
        prefill_list, decode_list = [], []
        for _ in range(MEASURE_RUNS):
            p, d = single_pass()
            prefill_list.append(p)
            decode_list.append(d)

        return {
            "prefill_ms" : float(np.mean(prefill_list)),
            "prefill_std": float(np.std(prefill_list)),
            "decode_ms"  : float(np.mean(decode_list)),
            "decode_std" : float(np.std(decode_list)),
        }

    results = []
    for il in INPUT_LENGTHS:
        log(f"Measuring input_len={il} ...")
        m = measure_latency(il)
        row = {"gpu_pct": gpu_pct, "input_len": il, **m}
        results.append(row)
        log(f"  prefill={m['prefill_ms']:.2f}ms  decode={m['decode_ms']:.3f}ms/tok")

    # 每行一个 JSON，orchestrator 按行解析
    for row in results:
        print(json.dumps(row), flush=True)


# ──────────────────────────── Orchestrator ────────────────────────────────
def orchestrate():
    """依次对每个 GPU% 启动 worker 子进程（MPS 已在外部启动）。"""
    all_results = []
    total = len(GPU_PERCENTAGES)

    for idx, pct in enumerate(GPU_PERCENTAGES, 1):
        print(f"\n{'═'*55}")
        print(f"  [{idx:2d}/{total}]  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = {pct}%")
        print(f"{'═'*55}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]             = "0"
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)

        cmd = [sys.executable, __file__, "--mode=worker"]
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=1200
            )
        except subprocess.TimeoutExpired:
            print(f"  [WARN] 超时，跳过 {pct}%")
            continue

        # 打印 worker stderr（进度日志）
        for line in proc.stderr.strip().splitlines():
            print(f"  {line}")

        # 解析 stdout 中的 JSON 行
        found = 0
        for line in proc.stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("{"):
                row = json.loads(line)
                all_results.append(row)
                found += 1
        if found == 0:
            print(f"  [WARN] 未收到结果，returncode={proc.returncode}")
            if proc.stdout: print(f"  stdout: {proc.stdout[-300:]}")
            if proc.stderr: print(f"  stderr: {proc.stderr[-300:]}")

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\n[Done] 结果已保存 → {RESULTS_FILE}  ({len(all_results)} 条)")
    return all_results


# ──────────────────────────── Plot ────────────────────────────────────────
def plot(results=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from collections import defaultdict

    if results is None:
        if not RESULTS_FILE.exists():
            print(f"[ERROR] 找不到 {RESULTS_FILE}")
            sys.exit(1)
        results = json.loads(RESULTS_FILE.read_text())

    # data[input_len][gpu_pct] = row
    data = defaultdict(dict)
    for r in results:
        data[r["input_len"]][r["gpu_pct"]] = r

    input_lens = sorted(data.keys())
    pcts       = sorted({r["gpu_pct"] for r in results})

    # 颜色：每条 input_len 一种颜色
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Knee GPU% — OPT-1.3b on Single V100 (CUDA MPS SM throttle)\n"
        f"decode={DECODE_STEPS} steps, {MEASURE_RUNS} runs per point (mean ± std)",
        fontsize=13, fontweight="bold"
    )

    # ── Left: Prefill ──────────────────────────────────────────────────
    for i, il in enumerate(input_lens):
        xs  = [p for p in pcts if p in data[il]]
        ys  = [data[il][p]["prefill_ms"]  for p in xs]
        es  = [data[il][p]["prefill_std"] for p in xs]
        ax1.errorbar(
            xs, ys, yerr=es,
            label=f"input len = {il} tokens",
            color=palette[i], marker=markers[i],
            linewidth=2.2, markersize=7, capsize=4, capthick=1.5,
        )
    ax1.set_xlabel("GPU SM Utilization (%)", fontsize=12)
    ax1.set_ylabel("Prefill Latency (ms)", fontsize=12)
    ax1.set_title("① Prefill Latency vs GPU%\n(compute-bound)", fontsize=12, fontweight="bold")
    ax1.set_xticks(pcts)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.35, linestyle="--")
    ax1.annotate(
        "Knee: curve flattens → extra SMs give\ndiminishing returns (memory-BW limited)",
        xy=(0.98, 0.97), xycoords="axes fraction",
        fontsize=8, color="gray", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
    )

    # ── Right: Decode ───────────────────────────────────────────────────
    for i, il in enumerate(input_lens):
        xs  = [p for p in pcts if p in data[il]]
        ys  = [data[il][p]["decode_ms"]  for p in xs]
        es  = [data[il][p]["decode_std"] for p in xs]
        ax2.errorbar(
            xs, ys, yerr=es,
            label=f"input len = {il} tokens",
            color=palette[i], marker=markers[i],
            linewidth=2.2, markersize=7, capsize=4, capthick=1.5,
        )
    ax2.set_xlabel("GPU SM Utilization (%)", fontsize=12)
    ax2.set_ylabel("Decode Latency (ms / token)", fontsize=12)
    ax2.set_title("② Decode Latency vs GPU%\n(memory-BW bound)", fontsize=12, fontweight="bold")
    ax2.set_xticks(pcts)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.35, linestyle="--")
    ax2.annotate(
        "Knee appears very early (~20% SM):\ndecode is memory-BW bound, not compute.\nSpikes at 30%/70%: cuBLAS kernel\nselection anomaly at those SM counts.",
        xy=(0.98, 0.97), xycoords="axes fraction",
        fontsize=8, color="gray", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved → {PLOT_FILE}")

    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "═"*72)
    print(f"{'GPU%':>5}  {'InLen':>6}  {'Prefill(ms)':>14}  {'Decode(ms/tok)':>16}")
    print("─"*72)
    for il in input_lens:
        for p in pcts:
            if p not in data[il]:
                continue
            r = data[il][p]
            print(f"{p:>5}%  {il:>6}  "
                  f"{r['prefill_ms']:>7.2f} ± {r['prefill_std']:<5.2f}  "
                  f"{r['decode_ms']:>8.3f} ± {r['decode_std']:.3f}")
        print()
    print("═"*72)


# ──────────────────────────── MPS 管理 ────────────────────────────────────
def ensure_mps_running():
    """检查 MPS daemon 是否已在 GPU 0 上运行，若无则尝试启动。"""
    out = subprocess.run(
        ["pgrep", "-x", "nvidia-cuda-mps"],
        capture_output=True, text=True
    )
    if out.returncode == 0:
        print("[MPS] daemon 已在运行。")
        return True
    print("[MPS] 正在启动 nvidia-cuda-mps-control -d ...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    r = subprocess.run(
        ["nvidia-cuda-mps-control", "-d"],
        env=env, capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"[WARN] MPS 启动失败（可能已在运行）: {r.stderr.strip()}")
        return False
    time.sleep(1)
    print("[MPS] 启动成功。")
    return True


# ──────────────────────────── Entry point ─────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="OPT-1.3b Knee GPU% via CUDA MPS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            模式说明:
              orchestrate  启动 MPS 并运行全部实验（默认）
              worker       单次测量（由 orchestrate 自动调用）
              plot         用已有 knee_results.json 重新绘图
        """),
    )
    parser.add_argument("--mode", default="orchestrate",
                        choices=["orchestrate", "worker", "plot"])
    args = parser.parse_args()

    if args.mode == "orchestrate":
        ensure_mps_running()
        results = orchestrate()
        plot(results)
    elif args.mode == "worker":
        worker(args)
    elif args.mode == "plot":
        plot()


if __name__ == "__main__":
    main()

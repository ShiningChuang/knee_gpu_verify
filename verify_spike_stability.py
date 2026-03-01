#!/usr/bin/env python3
"""
Spike Stability Test
====================
Run OPT-1.3B SM% sweep TWICE in the same MPS daemon session.
If spike position is MPS-state-dependent (not random):
  → same SM% should spike in both sweeps.
If truly random:
  → spike positions will differ between sweeps.
"""
import os, sys, json, subprocess
from pathlib import Path
import numpy as np

MODEL_ID        = "facebook/opt-1.3b"
GPU_PERCENTAGES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
N_SWEEPS        = 2
RESULTS_FILE    = Path(__file__).parent / "spike_stability_results.json"

WORKER_CODE = '''
import os, sys, json, time
import numpy as np, torch
from transformers import AutoModelForCausalLM

gpu_pct = int(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", 100))
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b", dtype=torch.float16,
    device_map={"": 0}, low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.eval()

ids    = torch.ones((1, 512), dtype=torch.long, device="cuda:0")
with torch.no_grad(): out = model(ids, use_cache=True)
past_kv = out.past_key_values
nt      = out.logits[:, -1:, :].argmax(-1)

DECODE_STEPS = 50
WARMUP       = 2
MEASURE      = 6

def one_decode():
    pk, tok = past_kv, nt
    steps = []
    for _ in range(DECODE_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(tok, past_key_values=pk, use_cache=True)
        torch.cuda.synchronize()
        steps.append((time.perf_counter()-t0)*1e3)
        pk, tok = out.past_key_values, out.logits[:,-1:,:].argmax(-1)
    return float(np.mean(steps))

for _ in range(WARMUP): one_decode()
dc = [one_decode() for _ in range(MEASURE)]
print(json.dumps({"gpu_pct": gpu_pct, "decode_ms": float(np.mean(dc)),
                  "decode_std": float(np.std(dc))}), flush=True)
'''

# Write worker to temp file
worker_path = Path(__file__).parent / "_stability_worker.py"
worker_path.write_text(WORKER_CODE)

all_results = []   # list of sweep results

for sweep in range(N_SWEEPS):
    print(f"\n{'━'*55}")
    print(f"  Sweep {sweep+1}/{N_SWEEPS}")
    print(f"{'━'*55}")
    sweep_results = []

    for pct in GPU_PERCENTAGES:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]             = "0"
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)
        proc = subprocess.run(
            [sys.executable, str(worker_path)],
            env=env, capture_output=True, text=True, timeout=600,
        )
        for line in proc.stderr.strip().splitlines():
            pass  # suppress weight-loading noise
        for line in proc.stdout.strip().splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                sweep_results.append(r)
                print(f"  SM={pct:3d}%  decode={r['decode_ms']:.3f}ms ± {r['decode_std']:.3f}ms")
                break
        else:
            print(f"  SM={pct:3d}%  [WARN] no result  rc={proc.returncode}")

    all_results.append(sweep_results)

RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
worker_path.unlink()

# ── Analysis ──────────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("Comparison: decode latency per SM%")
print(f"{'SM%':>5}  {'Sweep 1 (ms)':>14}  {'Sweep 2 (ms)':>14}  {'Δ (ms)':>8}  note")
print(f"{'─'*65}")

baselines = []
for sweep_res in all_results:
    b = next(r for r in sweep_res if r['gpu_pct']==100)
    baselines.append(b['decode_ms'])

for pct in GPU_PERCENTAGES:
    vals = []
    for sweep_res in all_results:
        r = next((x for x in sweep_res if x['gpu_pct']==pct), None)
        vals.append(r['decode_ms'] if r else float('nan'))

    ratios = [v/b for v, b in zip(vals, baselines)]
    is_spike = [r > 1.15 for r in ratios]
    note = ""
    if all(is_spike):
        note = "★ SPIKE in BOTH  → consistent"
    elif any(is_spike):
        note = "△ spike in one only → inconsistent"
    delta = vals[1] - vals[0] if len(vals) >= 2 else float('nan')
    print(f"{pct:>4}%  {vals[0]:>12.3f}ms  {vals[1]:>12.3f}ms  {delta:>+8.3f}  {note}")

print(f"{'─'*65}")
print(f"  Baseline (100% SM): Sweep1={baselines[0]:.3f}ms  Sweep2={baselines[1]:.3f}ms")
print(f"\nConclusion:")
spike_pcts_per_sweep = []
for i, sweep_res in enumerate(all_results):
    b = baselines[i]
    spikes = [r['gpu_pct'] for r in sweep_res if r['decode_ms']/b > 1.15]
    spike_pcts_per_sweep.append(spikes)
    print(f"  Sweep {i+1}: spike at SM% = {spikes or ['none']}")

if spike_pcts_per_sweep[0] == spike_pcts_per_sweep[1]:
    print("  → Spike positions MATCH  → MPS-state-dependent (structured)")
else:
    overlap = set(spike_pcts_per_sweep[0]) & set(spike_pcts_per_sweep[1])
    print(f"  → Spike positions DIFFER (overlap: {overlap or 'none'}) → position is not stable")
print(f"{'═'*65}")

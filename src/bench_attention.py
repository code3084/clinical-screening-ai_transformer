"""Task 4 validation: numerical agreement + honest speed/memory comparison.

python -m src.bench_attention            # writes results/attention/{accuracy,performance}.json
Ground truth = reference_attention in float64. Errors are max/mean |difference|.
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .attention import reference_attention, tiled_attention

SHAPES = [(1, 2, 16, 16, 8), (2, 4, 50, 50, 16), (1, 3, 37, 101, 32), (2, 2, 64, 64, 64),
          (1, 1, 1, 129, 64), (1, 8, 200, 200, 64), (1, 2, 130, 130, 24)]   # (B,H,Lq,Lk,D)
BLOCKS = [(16, 16), (64, 32), (128, 128)]


def sdpa(q, k, v, causal):
    mask = None
    if causal:
        Lq, Lk = q.shape[-2], k.shape[-2]
        mask = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool, device=q.device), diagonal=Lk - Lq)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def accuracy(dev):
    rows = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for (B, H, Lq, Lk, D) in SHAPES:
        q, k, v = (torch.randn(B, H, L, D, generator=g, dtype=torch.float64).to(dev) for L in (Lq, Lk, Lk))
        for causal in (True, False):
            truth = reference_attention(q, k, v, causal)
            cands = {"reference_fp32": reference_attention(q.float(), k.float(), v.float(), causal),
                     "sdpa_fp32": sdpa(q.float(), k.float(), v.float(), causal)}
            for bq, bk in BLOCKS:
                cands[f"tiled_fp32_{bq}x{bk}"] = tiled_attention(q.float(), k.float(), v.float(), causal, bq, bk)
                cands[f"tiled_fp64_{bq}x{bk}"] = tiled_attention(q, k, v, causal, bq, bk)
            for name, out in cands.items():
                e = (out.double() - truth).abs()
                rows.append(dict(shape=[B, H, Lq, Lk, D], causal=causal, impl=name,
                                 max_err=float(e.max()), mean_err=float(e.mean())))
    return rows


def timed(fn, dev, warm=2, reps=5):
    for _ in range(warm):
        fn()
    ts, peaks = [], []
    for _ in range(reps):
        if dev == "cuda":
            torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter(); out = fn()
        if dev == "cuda":
            torch.cuda.synchronize(); peaks.append((torch.cuda.max_memory_allocated() - base) / 2**20)
        ts.append(time.perf_counter() - t0)
    return out, 1000 * sum(ts) / len(ts), (max(peaks) if peaks else None)


def performance(dev, lengths=(256, 512, 1024, 2048, 4096)):
    dt = torch.float16 if dev == "cuda" else torch.float32
    rows = []
    for L in lengths:
        q, k, v = (torch.randn(1, 8, L, 64, device=dev, dtype=dt) for _ in range(3))
        impls = {"reference": lambda: reference_attention(q, k, v, True),
                 "tiled_128x128": lambda: tiled_attention(q, k, v, True, 128, 128),
                 "sdpa_default": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)}
        if dev == "cuda":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            for nm, be in (("sdpa_math", SDPBackend.MATH), ("sdpa_mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                           ("sdpa_flash", SDPBackend.FLASH_ATTENTION)):
                def f(be=be):
                    with sdpa_kernel(be):
                        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
                impls[nm] = f
        ref = None
        for name, fn in impls.items():
            try:
                out, ms, peak = timed(fn, dev)
            except Exception as e:                       # backend unsupported on this GPU
                rows.append(dict(L=L, impl=name, error=str(e)[:80])); continue
            ref = out if name == "reference" else ref
            rows.append(dict(L=L, impl=name, ms=ms, peak_extra_mb=peak, dtype=str(dt),
                             max_err_vs_reference=float((out.float() - ref.float()).abs().max())))
    return rows


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("results/attention"); out.mkdir(parents=True, exist_ok=True)
    acc = accuracy(dev)
    (out / "accuracy.json").write_text(json.dumps(acc, indent=1))
    worst = {}
    for r in acc:
        w = worst.setdefault(r["impl"], [0.0, 0.0]); w[0] = max(w[0], r["max_err"]); w[1] += r["mean_err"]
    print("worst-case max error vs float64 reference (all shapes, causal+non-causal):")
    for k_, (mx, _) in worst.items():
        print(f"  {k_:24s} {mx:.2e}")
    perf = performance(dev)
    (out / "performance.json").write_text(json.dumps(dict(device=dev, gpu=torch.cuda.get_device_name() if dev == "cuda" else None, rows=perf), indent=1))
    print(f"\ndevice={dev}  (ms per call, extra peak MB)")
    for r in perf:
        print(r if "error" in r else f"L={r['L']:5d} {r['impl']:20s} {r['ms']:9.2f} ms  peak+{r['peak_extra_mb'] or 0:8.1f} MB  err={r['max_err_vs_reference']:.1e}")

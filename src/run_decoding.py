"""Equivalence tests + benchmarks for KV-cache (Task 3) and speculative decoding (Task 5).

python -m src.run_decoding test  --adapter <HF id or runs/lora_r16/best>
python -m src.run_decoding bench --adapter <...> [--draft-adapter runs/draft_lora/best] [--dtype float16]
Adapter defaults to $TRIALSCREEN_ADAPTER, else runs/lora_r16/best.
Results: results/decoding/{equivalence,bench_kv,bench_spec}.json
"""
import argparse
import json
import os
from pathlib import Path

import torch

from .common import load_jsonl
from .decoding import (encode_prompt, forward, get_eos_ids, greedy_kvcache, greedy_nocache, hf_greedy,
                       kv_bytes_per_token, make_prompt_ids, speculative_greedy, _arange, _ones)

OUT = Path("results/decoding")


# ----------------------------------------------------------------------------- utilities
def mean_stats(runs):
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out = {k: sum(r[k] for r in runs) / len(runs) for k in keys}
    if len(runs) > 1:
        out["total_s_std"] = float(torch.tensor([r["total_s"] for r in runs]).std())
    return out


def timed(fn, warm=1, reps=3):
    """fn() -> (tokens, stats). Warm-up, then repeated synchronized runs; peak memory above resident weights."""
    for _ in range(warm):
        fn()
    runs, toks = [], None
    for _ in range(reps):
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize(); base = torch.cuda.memory_allocated() / 2**20; torch.cuda.reset_peak_memory_stats()
        toks, st = fn()
        if cuda:
            st["peak_extra_mb"] = st["peak_mem_mb"] - base
        runs.append(st)
    return toks, mean_stats(runs)


@torch.no_grad()
def top2_margin(model, ids, ref, idx):
    """Logit gap between best and 2nd-best token at generation step idx (teacher-forced on ref)."""
    full = torch.cat([ids, torch.tensor([ref[:idx]], device=ids.device, dtype=ids.dtype)], 1) if idx else ids
    L = full.shape[1]
    lg, _ = forward(model, full, mask=_ones(L, ids.device), pos=_arange(0, L, ids.device), use_cache=False)
    top = lg[0, -1].float().topk(2).values
    return float(top[0] - top[1])


@torch.no_grad()
def logit_drift(model, ids, ref):
    """Teacher-force `ref`; compare logits from the cached step-by-step path with one full forward."""
    dev, P, n = ids.device, ids.shape[1], len(ref)
    full = torch.cat([ids, torch.tensor([ref[:-1]], device=dev, dtype=ids.dtype)], 1)
    lf, _ = forward(model, full, mask=_ones(full.shape[1], dev), pos=_arange(0, full.shape[1], dev), use_cache=False, keep=n)
    lg, cache = forward(model, ids, mask=_ones(P, dev), pos=_arange(0, P, dev), keep=1)
    outs = [lg[:, -1]]
    for t in ref[:-1]:
        past = cache.get_seq_length()
        lg, cache = forward(model, torch.tensor([[t]], device=dev), mask=_ones(past + 1, dev),
                            pos=torch.tensor([[past]], device=dev), cache=cache, keep=1)
        outs.append(lg[:, -1])
    d = (torch.stack(outs, 1).float() - lf.float()).abs()
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()))


# ----------------------------------------------------------------------------- experiments
def run_equivalence(target, draft, cases, k, eos_default=None):
    rows = []
    for c in cases:
        ids, n, eos = c["ids"], c["n"], c["eos"]
        ref, _ = greedy_nocache(target, ids, n, eos)
        outs = {"kvcache": greedy_kvcache(target, ids, n, eos)[0],
                "hf_generate": hf_greedy(target, ids, n, eos)[0],
                "speculative": speculative_greedy(target, draft, ids, n, k=k, eos_ids=eos)[0]}
        try:
            outs["hf_assisted"] = hf_greedy(target, ids, n, eos, assistant=draft)[0]
        except Exception as e:                                  # library reference is optional
            outs["hf_assisted"] = None; c["hf_assisted_error"] = str(e)[:100]
        row = dict(case=c["name"], prompt_len=ids.shape[1], max_new=n, ref_len=len(ref))
        for name, o in outs.items():
            if o is None:
                row[f"{name}_equal"] = None; continue
            row[f"{name}_equal"] = o == ref
            if o != ref:
                i = next((j for j, (a, b) in enumerate(zip(o, ref)) if a != b), min(len(o), len(ref)))
                row[f"{name}_first_mismatch"] = i
                row[f"{name}_ref_top2_margin"] = top2_margin(target, ids, ref, min(i, len(ref) - 1))
        if c["name"].endswith("out64") and c["eos"] is None:
            row["logit_drift_cached_vs_full"] = logit_drift(target, ids, ref)
        rows.append(row)
        print({k_: v for k_, v in row.items() if k_.endswith("_equal") or k_ == "case"})
    return rows


def run_kv_bench(target, grid, warm, reps):
    bpt = kv_bytes_per_token(target.config, next(target.parameters()).dtype)
    rows = []
    for (L, N), ids in grid.items():
        ref, s_no = timed(lambda: greedy_nocache(target, ids, N, None), warm, reps)
        kv, s_kv = timed(lambda: greedy_kvcache(target, ids, N, None), warm, reps)
        row = dict(input_len=L, output_len=N, tokens_equal=kv == ref, kv_cache_theoretical_mb=(L + N) * bpt / 2**20,
                   nocache=s_no, kvcache=s_kv, decode_speedup=s_no["decode_s"] / s_kv["decode_s"],
                   total_speedup=s_no["total_s"] / s_kv["total_s"])
        rows.append(row)
        print(f"in={L} out={N} equal={row['tokens_equal']} decode_speedup={row['decode_speedup']:.1f}x "
              f"tok/s {s_no['decode_tok_per_s']:.1f} -> {s_kv['decode_tok_per_s']:.1f}")
    return rows


def run_spec_bench(target, draft, cases, ks, warm, reps):
    rows = []
    for c in cases:
        ids, n, eos = c["ids"], c["n"], c["eos"]
        base_toks, base = timed(lambda: greedy_kvcache(target, ids, n, eos), warm, reps)
        for k in ks:
            sp_toks, sp = timed(lambda: speculative_greedy(target, draft, ids, n, k=k, eos_ids=eos), warm, reps)
            row = dict(case=c["name"], prompt_len=ids.shape[1], max_new=n, k=k, tokens_equal=sp_toks == base_toks,
                       baseline=base, speculative=sp, speedup=base["total_s"] / sp["total_s"])
            if sp_toks != base_toks:                      # investigate: first divergence + target logit margin there
                i = next((j for j, (x, y) in enumerate(zip(sp_toks, base_toks)) if x != y), min(len(sp_toks), len(base_toks)) - 1)
                row.update(first_mismatch=i, ref_top2_margin=top2_margin(target, ids, base_toks, min(i, len(base_toks) - 1)))
                print("  MISMATCH", c["name"], "k=%d" % k, row["first_mismatch"], "margin=%.4f" % row["ref_top2_margin"])
            rows.append(row)
        print(c["name"], [(r["k"], round(r["speedup"], 2), round(r["speculative"]["acceptance_rate"], 2)) for r in rows if r["case"] == c["name"]])
    return rows


def predicted_speedup(alpha, k, c):
    """Decode-phase speedup model: expected tokens/round (1-a^(k+1))/(1-a) over cost k*c + 1 target-steps."""
    e = (k + 1) if alpha >= 1 else (1 - alpha ** (k + 1)) / (1 - alpha)
    return e / (k * c + 1)


def aggregate_task_rows(rows, ks, c=None):
    """Pool per-example rows of the real-task benchmark into one row per k."""
    out = []
    for k in ks:
        r = [x for x in rows if x["k"] == k]
        drafted, accepted = sum(x["speculative"]["drafted"] for x in r), sum(x["speculative"]["accepted"] for x in r)
        out.append(dict(k=k, n_examples=len(r), all_tokens_equal=all(x["tokens_equal"] for x in r),
                        acceptance_rate=accepted / drafted if drafted else 0.0,
                        mean_baseline_s=sum(x["baseline"]["total_s"] for x in r) / len(r),
                        mean_spec_s=sum(x["speculative"]["total_s"] for x in r) / len(r),
                        mean_baseline_tok_s=sum(x["baseline"]["e2e_tok_per_s"] for x in r) / len(r),
                        mean_spec_tok_s=sum(x["speculative"]["e2e_tok_per_s"] for x in r) / len(r),
                        mean_tokens_per_target_forward=sum(x["speculative"]["tokens_per_target_forward"] for x in r) / len(r),
                        speedup_of_means=(sum(x["baseline"]["total_s"] for x in r) / sum(x["speculative"]["total_s"] for x in r))))
        if c:
            out[-1]["predicted_decode_speedup"] = predicted_speedup(out[-1]["acceptance_rate"], k, c)
    return out


# ----------------------------------------------------------------------------- CLI
def load_all(a):
    from .model_utils import load_model, load_tokenizer
    dt = getattr(torch, a.dtype)
    tok, dtok = load_tokenizer(a.base), load_tokenizer(a.draft)
    assert tok.get_vocab() == dtok.get_vocab(), "draft/target tokenizers differ"
    target, draft = load_model(a.base, a.adapter, dt), load_model(a.draft, a.draft_adapter, dt)
    assert target.config.vocab_size == draft.config.vocab_size
    return tok, target, draft


def meta(a, target, draft):
    nb = lambda m: sum(p.numel() * p.element_size() for p in m.parameters()) / 2**20
    return dict(base=a.base, adapter=a.adapter, draft=a.draft, draft_adapter=a.draft_adapter, dtype=a.dtype,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, torch=torch.__version__,
                target_weights_mb=nb(target), draft_weights_mb=nb(draft),
                kv_bytes_per_token=kv_bytes_per_token(target.config, next(target.parameters()).dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test", "bench"])
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=os.environ.get("TRIALSCREEN_ADAPTER", "runs/lora_r16/best"))
    ap.add_argument("--draft-adapter", default=None)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--n-real", type=int, default=12, help="real validation prompts")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--outs", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tok, target, draft = load_all(a)
    dev, eos = next(target.parameters()).device, get_eos_ids(tok, target)
    val = load_jsonl("data/processed/val.jsonl")             # tests/benchmarks never touch the test split
    real = [dict(name=f"val{r['annotation_id']}", ids=encode_prompt(tok, r, dev), n=64, eos=eos)
            for r in val[::max(1, len(val) // a.n_real)][:a.n_real]]
    sfx = f"_{a.tag}" if a.tag else ""

    if a.cmd == "test":
        cases = [dict(name=f"in{L}_out{N}", ids=make_prompt_ids(tok, val, L, dev), n=N, eos=None)
                 for L in a.lengths for N in (16, 64, 128)] + real
        rows = run_equivalence(target, draft, cases, a.k)
        summary = {k_: sum(1 for r in rows if r.get(k_) is True) for k_ in
                   ("kvcache_equal", "hf_generate_equal", "speculative_equal", "hf_assisted_equal")}
        print("equal counts out of", len(rows), summary)
        (OUT / f"equivalence{sfx}.json").write_text(json.dumps(dict(meta=meta(a, target, draft), summary=summary, rows=rows), indent=1))
    else:
        grid = {(L, N): make_prompt_ids(tok, val, L, dev) for L in a.lengths for N in a.outs}
        kv_rows = run_kv_bench(target, grid, a.warm, a.reps)
        (OUT / f"bench_kv{sfx}.json").write_text(json.dumps(dict(meta=meta(a, target, draft), rows=kv_rows), indent=1))
        ids512 = make_prompt_ids(tok, val, 512, dev)
        _, s_t = timed(lambda: greedy_kvcache(target, ids512, 64, None), a.warm, a.reps)
        _, s_d = timed(lambda: greedy_kvcache(draft, ids512, 64, None), a.warm, a.reps)
        cost_ratio = s_t["decode_tok_per_s"] / s_d["decode_tok_per_s"]      # draft step time / target step time
        print(f"solo decode tok/s: target {s_t['decode_tok_per_s']:.1f} draft {s_d['decode_tok_per_s']:.1f} -> cost ratio c={cost_ratio:.2f}")
        task_rows = run_spec_bench(target, draft, real, a.ks, a.warm, 1)
        fixed = [dict(name=f"fixed_in512_out{N}", ids=make_prompt_ids(tok, val, 512, dev), n=N, eos=None) for N in (64, 128)]
        fixed_rows = run_spec_bench(target, draft, fixed, a.ks, a.warm, a.reps)
        (OUT / f"bench_spec{sfx}.json").write_text(json.dumps(
            dict(meta=meta(a, target, draft), solo=dict(target=s_t, draft=s_d, cost_ratio=cost_ratio),
                 task_summary=aggregate_task_rows(task_rows, a.ks, cost_ratio),
                 task_rows=task_rows, fixed_rows=fixed_rows), indent=1))
        for r in aggregate_task_rows(task_rows, a.ks, cost_ratio):
            print("task k=%(k)d acc=%(acceptance_rate).2f speedup=%(speedup_of_means).2f equal=%(all_tokens_equal)s predicted_decode_speedup=%(predicted_decode_speedup).2f" % r)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

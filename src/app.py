"""Browser application (Gradio).  Start:  python -m src.app   ->  http://localhost:7860

Tabs: Screening (criterion-level results + evidence + report export) | Inference benchmark (live comparison of
no-cache / KV-cache / speculative decoding) | Model & held-out results | About.
"""
import argparse
import json
import os
import time
from pathlib import Path

import gradio as gr
from matplotlib.figure import Figure

from .examples import EXAMPLES
from .screening import (DISCLAIMER, AGGREGATION_RULE, aggregate, build_result, cards_html, make_meta, overall_html,
                        parse_criteria, prepare_inputs, report_html, report_markdown, segment_note, esc)

VARIANTS = {"Fine-tuned LoRA (recommended)": "finetuned", "Base model, zero-shot (for comparison)": "base"}
MODES = {"KV cache (default)": "kvcache", "No cache (reference)": "nocache", "Speculative decoding (draft + verify)": "speculative"}
REPORT_DIR = Path("reports")
BANNER = f'<div class="ts-banner"><b>{esc(DISCLAIMER)}</b></div>'
TABLE_HEADERS = ["#", "Type", "Criterion", "Prediction", "Cited sentence IDs", "Seconds"]


# ----------------------------------------------------------------------------- handlers (testable, no server needed)
def screen_handler(engine, note, crit_text, default_type, variant_label, mode_label, k, progress=None):
    variant, mode, k = VARIANTS[variant_label], MODES[mode_label], int(k)
    try:
        sents, crits, warns = prepare_inputs(note, crit_text, default_type, engine.count_tokens)
    except ValueError as e:                                   # invalid input -> friendly message, app keeps running
        raise gr.Error(str(e))
    try:
        engine.ensure_ready(variant, mode)
    except Exception as e:
        raise gr.Error(f"Could not load the model: {e}")
    t0, results = time.time(), []
    for n, c in enumerate(crits):
        if progress is not None:
            progress((n, len(crits)), desc=f"Criterion {n + 1} of {len(crits)}")
        try:
            pred = engine.predict(sents, c["type"], c["text"], variant, mode, k)
        except Exception as e:                                # one failing criterion must not lose the others
            pred = dict(label=None, evidence=[], valid=False, raw=f"ERROR: {type(e).__name__}: {e}", stats={},
                        prompt_tokens=c.get("prompt_tokens"))
        results.append(build_result(c, pred, sents))
    secs = time.time() - t0
    overall = aggregate(results)
    meta = make_meta(engine.info(), variant_label, mode_label, k, secs)
    REPORT_DIR.mkdir(exist_ok=True)
    stem = REPORT_DIR / f"trialscreen_report_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    paths = []
    for ext, fn in (("md", report_markdown), (".html".lstrip("."), report_html)):
        p = f"{stem}.{ext}"
        Path(p).write_text(fn(meta, sents, results, overall), encoding="utf-8")
        paths.append(p)
    table = [[r["index"], r["type"], r["text"][:90], r["label"] or "UNPARSABLE",
              ", ".join(map(str, r["evidence"])) or "-", round(r["stats"].get("total_s", 0.0), 2)] for r in results]
    failed = sum(1 for r in results if r["raw"].startswith("ERROR"))
    status = (f"Screened {len(results)} criteria against {len(sents)} patient sentences in {secs:.1f}s "
              f"({variant_label}; {mode_label}; {engine.info()['device']}). Reports saved to `{REPORT_DIR}/`."
              + (f"\n\n**{failed} criteria failed during inference and are marked unparsable.**" if failed else "")
              + "".join(f"\n\n> {w}" for w in warns))
    return status, overall_html(overall), table, cards_html(results, sents), paths


def make_bench_figure(rows):
    fig = Figure(figsize=(9, 3.2))
    a1, a2 = fig.subplots(1, 2)
    names = [r["mode"].replace(" (reference)", "\n(reference)") for r in rows]
    a1.bar(names, [r["total_s"] for r in rows], color="#4b7bec"); a1.set_ylabel("end-to-end latency (s)")
    a2.bar(names, [r["decode_tok_per_s"] for r in rows], color="#20bf6b"); a2.set_ylabel("decode tokens / s")
    for a in (a1, a2):
        a.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


def bench_handler(engine, note, crit_text, default_type, variant_label, k, reps, fixed, plen, nnew):
    used_example = not (note or "").strip() and not (crit_text or "").strip()
    if used_example:
        ex = next(iter(EXAMPLES.values())); note, crit_text = ex["note"], ex["criteria"]
    try:
        sents = segment_note(note)
        crit = parse_criteria(crit_text, default_type)[0]
    except ValueError as e:
        raise gr.Error(str(e))
    reps, plen, nnew, msg = int(reps), int(plen), int(nnew), []
    if engine.info()["device"] == "cpu":                      # keep CPU runs short
        reps, plen, nnew = 1, min(plen, 256), min(nnew, 32)
        msg.append("CPU detected: repetitions, prompt length and output length were capped to keep the run short.")
    try:
        engine.ensure_ready(VARIANTS[variant_label], "speculative")
        rows = engine.bench(sents, crit["type"], crit["text"], VARIANTS[variant_label], int(k), reps,
                            plen if fixed else None, nnew)
    except Exception as e:
        raise gr.Error(f"Benchmark failed: {type(e).__name__}: {e}")
    kv = next(r for r in rows if r["mode"] == "KV cache")
    table = [[r["mode"], r["new_tokens"], "yes" if r["equal_to_reference"] else "NO", round(r["prefill_s"], 3),
              round(r["decode_s"], 3), round(r["total_s"], 3), round(r["decode_tok_per_s"], 1),
              round(r["e2e_tok_per_s"], 1), "n/a" if r["peak_extra_mb"] is None else round(r["peak_extra_mb"], 1),
              "-" if r["acceptance_rate"] is None else f"{r['acceptance_rate']:.2f}",
              round(kv["total_s"] / r["total_s"], 2)] for r in rows]
    bad = [r["mode"] for r in rows if not r["equal_to_reference"]]
    verdict = ("All decoding paths produced **identical tokens** to the uncached reference." if not bad else
               f"Token mismatch vs the uncached reference for: {', '.join(bad)} (see the report: near-tied logits can flip in fp16).")
    setting = (f"fixed length: {rows[0]['prompt_len']}-token prompt, {nnew} new tokens (EOS ignored)" if fixed else
               f"task prompt of {rows[0]['prompt_len']} tokens: criterion '{crit['text'][:60]}' (stops at EOS)")
    status = (f"**Benchmark** - {setting}; {reps} timed repetition(s) after 1 warm-up. {verdict} "
              + ("Used the built-in example. " if used_example else "") + " ".join(msg))
    return status, table, make_bench_figure(rows)


def load_results_table(root="."):
    rows, names = [], {"majority": "Majority-class baseline", "base_zeroshot": "Base model, zero-shot",
                       "lora_r16": "LoRA r=16 (library-default decoding)", "lora_r16_purgreedy": "LoRA r=16 (pure greedy)"}
    for run, label in names.items():
        for split in ("val", "test"):
            p = Path(root) / "results" / run / f"{split}_metrics.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text())
            g = lambda s, k: round(m[s]["classification"][k], 3)
            ev = m["overall"]["evidence"]
            rows.append([label, split, g("overall", "accuracy"), g("overall", "macro_f1"), g("inclusion", "accuracy"),
                         g("inclusion", "macro_f1"), g("exclusion", "accuracy"), g("exclusion", "macro_f1"),
                         round(ev["all"]["micro_f1"], 3), round(ev["gold_nonempty"]["micro_f1"], 3)])
    return rows


RESULT_HEADERS = ["Run", "Split", "Acc", "Macro-F1", "Incl acc", "Incl F1", "Excl acc", "Excl F1",
                  "Evidence micro-F1", "Evidence micro-F1 (gold non-empty)"]


# ----------------------------------------------------------------------------- UI
def build_app(engine):
    info = engine.info()
    with gr.Blocks(title="TrialScreen - clinical trial eligibility screening (educational prototype)") as demo:
        gr.Markdown("# TrialScreen\nCriterion-level clinical-trial eligibility screening with a fine-tuned open-weight "
                    f"Transformer ({esc(info['base'])} + LoRA). Running on **{info['device']}**.")
        gr.HTML(BANNER)
        with gr.Tabs():
            with gr.Tab("Screening"):
                example = gr.Dropdown(list(EXAMPLES), value=None, label="Load a synthetic example (optional)")
                with gr.Row():
                    note = gr.Textbox(label="Patient record (free text; no real identifiers)", lines=12,
                                      placeholder="Paste a clinical note. Sentences are numbered automatically.")
                    crit = gr.Textbox(label="Trial eligibility criteria", lines=12,
                                      placeholder="Inclusion Criteria:\n- Age 18 or older\n\nExclusion Criteria:\n- Pregnant or breastfeeding\n"
                                                  "(headings switch the type; one criterion per bullet/line; indent to continue a line)")
                with gr.Accordion("Model and decoding options", open=False):
                    variant = gr.Radio(list(VARIANTS), value=list(VARIANTS)[0], label="Model")
                    mode = gr.Radio(list(MODES), value=list(MODES)[0], label="Decoding path (outputs are identical; speed differs)")
                    k = gr.Slider(1, 8, value=4, step=1, label="Speculative draft length k")
                    dtype = gr.Radio(["inclusion", "exclusion"], value="inclusion",
                                     label="Criterion type for lines that have no 'Inclusion/Exclusion Criteria' heading")
                run = gr.Button("Run screening", variant="primary")
                status = gr.Markdown()
                overall = gr.HTML()
                table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False, wrap=True, label="Criterion-level summary")
                details = gr.HTML()
                files = gr.File(label="Download screening report (Markdown + printable HTML)", file_count="multiple")

                def _load(name):
                    ex = EXAMPLES.get(name)
                    return (ex["note"], ex["criteria"]) if ex else (gr.update(), gr.update())

                def _screen(n, c, d, v, m, kk, progress=gr.Progress()):
                    return screen_handler(engine, n, c, d, v, m, kk, progress)

                example.change(_load, example, [note, crit], api_name="load_example")
                run.click(_screen, [note, crit, dtype, variant, mode, k], [status, overall, table, details, files], api_name="screen")

            with gr.Tab("Inference benchmark"):
                gr.Markdown("Runs the **same prompt** through three hand-implemented decoding paths - no cache, KV cache, and greedy "
                            "speculative decoding - and verifies that all produce identical tokens. Uses the first criterion from the "
                            "Screening tab (or a built-in example if those boxes are empty).")
                with gr.Row():
                    bk = gr.Slider(1, 8, value=4, step=1, label="Speculative k")
                    breps = gr.Slider(1, 5, value=3, step=1, label="Timed repetitions")
                    fixed = gr.Checkbox(value=False, label="Fixed-length mode (tile prompt, ignore EOS)")
                with gr.Row():
                    plen = gr.Slider(128, 1024, value=512, step=64, label="Prompt length (fixed mode)")
                    nnew = gr.Slider(16, 128, value=64, step=16, label="New tokens (fixed mode)")
                bbtn = gr.Button("Run decoding comparison", variant="primary")
                bstatus = gr.Markdown()
                btable = gr.Dataframe(headers=["Path", "New tokens", "Same tokens as reference", "Prefill s", "Decode s", "Total s",
                                               "Decode tok/s", "End-to-end tok/s", "Extra peak MB", "Draft acceptance",
                                               "Speed-up vs KV cache"], interactive=False, wrap=True)
                bplot = gr.Plot()

                def _bench(n, c, d, v, kk, r, f, pl, nn):
                    return bench_handler(engine, n, c, d, v, kk, r, f, pl, nn)

                bbtn.click(_bench, [note, crit, dtype, variant, bk, breps, fixed, plen, nnew], [bstatus, btable, bplot], api_name="benchmark")

            with gr.Tab("Model and held-out results"):
                gr.Markdown(f"**Base:** `{info['base']}` &nbsp; **Adapter:** `{info['adapter']}` &nbsp; **Draft:** `{info['draft']}`. "
                            "Held-out numbers below are read from the saved evaluation files (patient-grouped split, seed 42).")
                gr.Dataframe(value=load_results_table(), headers=RESULT_HEADERS, interactive=False, wrap=True)
                curves = Path("runs/lora_r16/curves.png")
                if curves.exists():
                    gr.Image(value=str(curves), label="Fine-tuning curves", interactive=False)

            with gr.Tab("About and limitations"):
                gr.Markdown(f"""
**Intended use.** Educational demonstration of criterion-level eligibility screening. It is **not** a medical device and must not
guide enrollment decisions.

**How an overall outcome is formed.** The dataset only provides criterion-level labels, so the overall outcome is an explicit
aggregation rule, not a ground-truth label: {AGGREGATION_RULE}

**Explanations** are templates built from the model's label and the sentences it cites; the model does not write free-text reasoning.

**Known limitations.** Fine-tuned on 37 patients; 'excluded' was very rare in training (8 examples) so the model mostly predicts
'not excluded' for exclusion criteria; cited evidence can be wrong; sentence splitting is rule-based; notes longer than ~580 tokens were
never seen in training. Do not enter real patient identifiers - inputs are processed on the server and saved in generated reports.
""")
    return demo


def make_engine(argv=None):
    from .engine import Engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter"); ap.add_argument("--draft-adapter"); ap.add_argument("--base"); ap.add_argument("--draft")
    ap.add_argument("--device"); ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--share", action="store_true"); ap.add_argument("--preload", action="store_true")
    a = ap.parse_args(argv)
    eng = Engine(a.base, a.adapter, a.draft, a.draft_adapter, a.device)
    if a.preload:
        eng.ensure_ready("finetuned", "kvcache")
    return eng, a


def main():
    eng, a = make_engine()
    build_app(eng).queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()

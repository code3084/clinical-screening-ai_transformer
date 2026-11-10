"""Pure-Python screening logic (no torch / gradio): input validation, sentence segmentation,
criteria parsing, rule-based aggregation, templated explanations, HTML / Markdown reports."""
import datetime as dt
import html
import re
from collections import Counter

from .common import split_note

APP_VERSION = "1.0"
DISCLAIMER = ("EDUCATIONAL SCREENING PROTOTYPE - NOT a clinical enrollment decision. Outputs come from a small "
              "fine-tuned language model, can be wrong, and must be reviewed by qualified clinical staff. "
              "Do not enter real patient identifiers.")
MAX_NOTE_CHARS, MAX_SENTENCES, MAX_CRITERIA, MAX_CRITERION_CHARS = 12000, 60, 20, 1500
WARN_PROMPT_TOKENS, MAX_PROMPT_TOKENS = 700, 1800     # longest training example was 580 tokens
AGGREGATION_RULE = (
    "(1) any exclusion criterion labelled 'excluded' -> likely not eligible; "
    "(2) otherwise any inclusion criterion labelled 'not included' -> likely not eligible; "
    "(3) otherwise any criterion labelled 'not enough information' (or with unparsable model output) -> "
    "undetermined, more information needed; (4) otherwise -> potentially eligible (preliminary). "
    "'not applicable' criteria are ignored. This is an explicit rule over criterion-level predictions, "
    "not a ground-truth label, and absence of evidence is never treated as proof of eligibility.")
LABEL_KIND = {"included": "ok", "not excluded": "ok", "not included": "bad", "excluded": "bad",
              "not enough information": "unk", "not applicable": "na"}

# ----------------------------------------------------------------------------- input handling
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_HEAD = re.compile(r"^\s*(?:#+\s*)?(inclusion|exclusion)\s+criteria\s*:?\s*$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


def segment_note(text, max_sentences=MAX_SENTENCES):
    """Free text -> list of sentences (index == sentence ID). Dataset-style '0. ... 1. ...' notes are respected."""
    t = (text or "").strip()
    if not t:
        raise ValueError("The patient record is empty. Paste a clinical note or load an example.")
    if len(t) > MAX_NOTE_CHARS:
        raise ValueError(f"The patient record is too long ({len(t)} characters; limit {MAX_NOTE_CHARS}).")
    if re.match(r"^0\.\s", t) and re.search(r"(?<!\S)1\.\s", t):
        sents = split_note(t)
    else:
        sents = []
        for line in t.splitlines():
            if line.strip():
                sents += _SENT_SPLIT.split(line.strip())
    sents = [re.sub(r"\s+", " ", s).strip() for s in sents if s.strip()]
    if not sents:
        raise ValueError("No sentences could be found in the patient record.")
    if len(sents) > max_sentences:
        raise ValueError(f"The patient record has {len(sents)} sentences; the limit is {max_sentences}.")
    return sents


def parse_criteria(text, default_type="inclusion"):
    """'Inclusion Criteria:' / 'Exclusion Criteria:' headings switch type; bullets or lines are criteria;
    an indented non-bullet line continues the previous criterion."""
    if default_type not in ("inclusion", "exclusion"):
        raise ValueError("Default criterion type must be 'inclusion' or 'exclusion'.")
    if not (text or "").strip():
        raise ValueError("No eligibility criteria were entered. Paste criteria or load an example.")
    ctype, items, cur = default_type, [], None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            cur = None
            continue
        m = _HEAD.match(line)
        if m:
            ctype, cur = m.group(1).lower(), None
        elif _BULLET.match(line):
            cur = dict(type=ctype, text=_BULLET.sub("", line, count=1).strip())
            items.append(cur)
        elif cur is not None and raw[:1] in (" ", "\t"):
            cur["text"] += " " + line.strip()
        else:
            cur = dict(type=ctype, text=line.strip())
            items.append(cur)
    items = [c for c in items if c["text"]]
    if not items:
        raise ValueError("No criteria were found in the text.")
    if len(items) > MAX_CRITERIA:
        raise ValueError(f"{len(items)} criteria entered; the limit is {MAX_CRITERIA} (CPU inference is slow).")
    for i, c in enumerate(items, 1):
        c["index"] = i
        if len(c["text"]) > MAX_CRITERION_CHARS:
            raise ValueError(f"Criterion #{i} is too long ({len(c['text'])} characters; limit {MAX_CRITERION_CHARS}).")
    return items


def prepare_inputs(note, criteria_text, default_type, count_tokens):
    """Validate everything BEFORE any model call. Returns (sentences, criteria, warnings)."""
    sents = segment_note(note)
    crits = parse_criteria(criteria_text, default_type)
    warns, worst = [], 0
    for c in crits:
        c["prompt_tokens"] = count_tokens(sents, c["type"], c["text"])
        worst = max(worst, c["prompt_tokens"])
    if worst > MAX_PROMPT_TOKENS:
        raise ValueError(f"The prompt would be {worst} tokens (limit {MAX_PROMPT_TOKENS}); shorten the record.")
    if worst > WARN_PROMPT_TOKENS:
        warns.append(f"Prompt length {worst} tokens exceeds the longest training example (580); accuracy may degrade.")
    return sents, crits, warns

# ----------------------------------------------------------------------------- results
def explain(r, sentences):
    """Template explanation built ONLY from the model's label and its cited sentences."""
    if not r["valid"]:
        return "The model output could not be parsed into a valid label, so this criterion is treated as undetermined."
    ev, lab = r["evidence"], r["label"]
    cited = " ".join(f"[{i}]" for i in ev)
    base = {
        "included": "Judged to MEET this inclusion criterion",
        "not included": "Judged NOT to meet this inclusion criterion",
        "excluded": "Judged to MEET this exclusion criterion (the patient would be excluded)",
        "not excluded": "Judged NOT to meet this exclusion criterion",
        "not enough information": "The record does not appear to contain enough information to decide",
        "not applicable": "Judged not applicable to this patient"}[lab]
    s = base + (f", citing sentence(s) {cited}." if ev else ". No patient sentence was cited.")
    if not ev and lab in ("not excluded", "not enough information"):
        s += " Absence of a mention is not proof that the condition is absent."
    return s + " (Template text; the model itself outputs only a label and sentence IDs.)"


def build_result(crit, pred, sentences):
    r = dict(index=crit["index"], type=crit["type"], text=crit["text"], label=pred["label"], valid=pred["valid"],
             evidence=list(pred["evidence"]), raw=pred.get("raw", ""), stats=pred.get("stats", {}),
             prompt_tokens=pred.get("prompt_tokens"))
    r["evidence_text"] = [(i, sentences[i]) for i in r["evidence"]]
    r["explanation"] = explain(r, sentences)
    return r


def aggregate(results):
    def ids(rs):
        return ", ".join(f"#{r['index']}" for r in rs)
    ok = lambda r, lab: r["valid"] and r["label"] == lab
    excl = [r for r in results if ok(r, "excluded")]
    notinc = [r for r in results if ok(r, "not included")]
    unk = [r for r in results if (not r["valid"]) or r["label"] == "not enough information"]
    if excl:
        code, title, why = "likely_not_eligible", f"Likely NOT eligible - exclusion criteria met: {ids(excl)}", excl
    elif notinc:
        code, title, why = "likely_not_eligible", f"Likely NOT eligible - inclusion criteria not met: {ids(notinc)}", notinc
    elif unk:
        code, title, why = "undetermined", f"Undetermined - more information needed for criteria: {ids(unk)}", unk
    else:
        code, title, why = ("potentially_eligible",
                            "Potentially eligible (preliminary) - no exclusion met and all applicable inclusion criteria met", [])
    return dict(code=code, title=title, reasons=[f"#{r['index']} ({r['type']}): {r['text']}" for r in why],
                counts=dict(Counter(r["label"] or "unparsable" for r in results)), rule=AGGREGATION_RULE)

# ----------------------------------------------------------------------------- rendering
CSS = """<style>
.ts-banner{border:2px solid #d97706;background:rgba(217,119,6,.12);padding:10px 14px;border-radius:8px;margin:8px 0}
.ts-overall{padding:12px 16px;border-radius:8px;margin:10px 0;font-size:1.05em;border:1px solid rgba(128,128,128,.4)}
.ts-overall.likely_not_eligible{background:rgba(220,38,38,.14)} .ts-overall.undetermined{background:rgba(217,119,6,.14)}
.ts-overall.potentially_eligible{background:rgba(22,163,74,.14)}
.ts-card{border:1px solid rgba(128,128,128,.4);border-radius:8px;padding:10px 14px;margin:10px 0}
.ts-badge{display:inline-block;padding:2px 10px;border-radius:12px;font-weight:600;font-size:.9em}
.ts-badge.ok{background:rgba(22,163,74,.25)} .ts-badge.bad{background:rgba(220,38,38,.25)}
.ts-badge.unk{background:rgba(217,119,6,.28)} .ts-badge.na{background:rgba(128,128,128,.25)}
.ts-type{font-size:.8em;text-transform:uppercase;opacity:.7;margin-right:6px}
.ts-sent{padding:2px 6px;margin:2px 0;border-radius:4px} .ts-cited{background:rgba(250,204,21,.35);font-weight:600}
.ts-muted{opacity:.7;font-size:.9em} .ts-card details{margin-top:6px} .ts-card ul{margin:4px 0 4px 18px}
</style>"""


def esc(x):
    return html.escape(str(x), quote=True)


def _badge(r):
    return f'<span class="ts-badge {LABEL_KIND.get(r["label"], "unk")}">{esc(r["label"] or "UNPARSABLE OUTPUT")}</span>'


def note_html(sentences, cited):
    return "".join(f'<div class="ts-sent{" ts-cited" if i in cited else ""}"><b>[{i}]</b> {esc(s)}</div>'
                   for i, s in enumerate(sentences))


def overall_html(overall):
    reasons = "".join(f"<li>{esc(x)}</li>" for x in overall["reasons"])
    counts = ", ".join(f"{esc(k)}: {v}" for k, v in sorted(overall["counts"].items()))
    return (CSS + f'<div class="ts-overall {overall["code"]}"><b>Preliminary overall outcome:</b> {esc(overall["title"])}'
            f'{"<ul>" + reasons + "</ul>" if reasons else ""}<div class="ts-muted">Label counts - {counts}</div>'
            f'<details><summary class="ts-muted">How this outcome is derived</summary><p class="ts-muted">{esc(overall["rule"])}</p></details></div>')


def cards_html(results, sentences):
    out = [CSS]
    for r in results:
        if r["evidence_text"]:
            ev = "<ul>" + "".join(f"<li><b>[{i}]</b> {esc(t)}</li>" for i, t in r["evidence_text"]) + "</ul>"
        else:
            ev = '<p class="ts-muted">No patient sentence cited.</p>'
        st = r["stats"] or {}
        timing = f' &middot; {st["total_s"]:.2f}s, {st.get("new_tokens", 0)} tokens' if "total_s" in st else ""
        out.append(
            f'<div class="ts-card"><div><span class="ts-type">#{r["index"]} {esc(r["type"])}</span>{_badge(r)}'
            f'<span class="ts-muted">{timing}</span></div><p><b>{esc(r["text"])}</b></p>'
            f'<div><i>Supporting evidence (exact patient sentences):</i>{ev}</div><p>{esc(r["explanation"])}</p>'
            f'<details><summary>Patient record with evidence highlighted</summary>{note_html(sentences, set(r["evidence"]))}</details>'
            f'<details><summary>Raw model output</summary><pre>{esc(r["raw"])}</pre></details></div>')
    return "".join(out)


def _meta_rows(meta):
    keys = [("Generated (UTC)", "timestamp"), ("Model", "variant"), ("Base model", "base"), ("LoRA adapter", "adapter"),
            ("Decoding path", "mode"), ("Device / dtype", "device"), ("Total inference time (s)", "seconds"),
            ("Prompt format", "prompt_format"), ("Application version", "app_version")]
    return [(label, meta.get(k, "")) for label, k in keys]


def report_markdown(meta, sentences, results, overall):
    L = ["# TrialScreen eligibility screening report", "", f"> **{DISCLAIMER}**", "", "## Provenance", "",
         "| Field | Value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in _meta_rows(meta)]
    L += ["", "## Preliminary overall outcome", "", f"**{overall['title']}**", ""]
    L += [f"- {x}" for x in overall["reasons"]] + ["", f"_Rule:_ {overall['rule']}", "", "## Criterion-level results", ""]
    for r in results:
        L += [f"### #{r['index']} ({r['type']}) {r['text']}", "", f"- **Prediction:** {r['label'] or 'UNPARSABLE OUTPUT'}",
              f"- **Cited sentence IDs:** {', '.join(map(str, r['evidence'])) or 'none'}"]
        L += [f"  - [{i}] {t}" for i, t in r["evidence_text"]]
        L += [f"- **Explanation:** {r['explanation']}", f"- **Raw model output:** `{r['raw'].strip()[:300]}`", ""]
    L += ["## Patient record as processed (sentence IDs)", ""] + [f"[{i}] {s}" for i, s in enumerate(sentences)]
    L += ["", "## Limitations", "", "Small model fine-tuned on 37 patients; expert-labelled exclusion criteria were rare in training; "
          "evidence sentences are model-selected and unverified. Not for clinical use."]
    return "\n".join(L)


def report_html(meta, sentences, results, overall):
    rows = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in _meta_rows(meta))
    return ("<!doctype html><html><head><meta charset='utf-8'><title>TrialScreen report</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 12px}"
            "table{border-collapse:collapse}td{border:1px solid #bbb;padding:4px 10px}</style></head><body>"
            f"<h1>TrialScreen eligibility screening report</h1><div class='ts-banner'><b>{esc(DISCLAIMER)}</b></div>"
            f"<h2>Provenance</h2><table>{rows}</table><h2>Overall</h2>{overall_html(overall)}"
            f"<h2>Criterion-level results</h2>{cards_html(results, sentences)}"
            f"<h2>Patient record as processed</h2>{note_html(sentences, set())}</body></html>")


def make_meta(info, variant_label, mode_label, k, seconds):
    return dict(timestamp=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), variant=variant_label,
                base=info.get("base"), adapter=info.get("adapter") if "Fine" in variant_label else "none (base weights)",
                mode=mode_label + (f", k={k}" if "pecul" in mode_label else ""),
                device=f"{info.get('device')} / {info.get('dtype')}", seconds=f"{seconds:.1f}",
                prompt_format="criterion-level JSON v1 (src/common.py)", app_version=APP_VERSION)

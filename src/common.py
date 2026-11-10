"""Shared constants, parsing and prompt/target formatting.

Used by data prep, training, evaluation, decoding benchmarks and the app, so
the prompt format is defined in exactly ONE place.
"""
import ast
import json
import re

INCLUSION_LABELS = ["included", "not included", "not enough information", "not applicable"]
EXCLUSION_LABELS = ["excluded", "not excluded", "not enough information", "not applicable"]
LABELS_BY_TYPE = {"inclusion": INCLUSION_LABELS, "exclusion": EXCLUSION_LABELS}
ALL_LABELS = sorted(set(INCLUSION_LABELS + EXCLUSION_LABELS))

# Columns that must NEVER reach prompts, targets or evaluation inputs.
FORBIDDEN_COLUMNS = {"gpt4_explanation", "explanation_correctness", "gpt4_sentences",
                     "gpt4_eligibility", "training"}


def parse_id_list(x):
    """'[0, 1, 2]' | list | None -> sorted list[int]."""
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return []
        x = ast.literal_eval(s)
    try:
        return sorted({int(i) for i in list(x)})
    except TypeError:
        return [int(x)]


def split_note(note: str):
    """Split '0. A. 1. B. 2. C.' into ['A.', 'B.', 'C.'] (list index == sentence ID).

    Looks for markers 0., 1., 2., ... strictly in sequence, so numbers inside a
    sentence ("1.5 cm", "grade 3.") are only mistaken for a marker in rare
    cases; data_prep verifies every gold ID fits the resulting sentence count.
    """
    note = note.strip()
    marks, pos, k = [], 0, 0
    while True:
        m = re.compile(rf"(?<!\S){k}\.\s").search(note, pos)
        if m is None:
            break
        marks.append((m.start(), m.end()))
        pos, k = m.end(), k + 1
    if not marks:
        return [note]
    sents = []
    for i, (_, end) in enumerate(marks):
        nxt = marks[i + 1][0] if i + 1 < len(marks) else len(note)
        sents.append(note[end:nxt].strip())
    return sents


def build_user_prompt(sentences, criterion_type: str, criterion_text: str) -> str:
    labels = ", ".join(f'"{l}"' for l in LABELS_BY_TYPE[criterion_type])
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return (
        "You are screening a patient against ONE clinical-trial eligibility criterion. "
        "Use only the patient note. Absence of information is not proof either way.\n\n"
        f"### Patient note (numbered sentences)\n{numbered}\n\n"
        f"### Criterion type\n{criterion_type}\n\n"
        f"### Criterion\n{criterion_text.strip()}\n\n"
        f"### Allowed labels\n{labels}\n\n"
        'Answer with JSON only: {"label": <one allowed label>, "evidence": [<sentence ids>]}. '
        "Use an empty list when no sentence is relevant."
    )


def build_target(label: str, evidence_ids) -> str:
    return json.dumps({"label": label, "evidence": [int(i) for i in evidence_ids]})


_JSON_RE = re.compile(r"\{.*?\}", re.S)


def parse_output(text: str, criterion_type: str, n_sentences: int):
    """Parse model output -> (label|None, evidence_ids, is_valid). Never raises."""
    for m in _JSON_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        label = obj.get("label")
        ev = obj.get("evidence", [])
        if label not in LABELS_BY_TYPE[criterion_type] or not isinstance(ev, list):
            continue
        try:
            ids = sorted({int(i) for i in ev if 0 <= int(i) < n_sentences})
        except (TypeError, ValueError):
            continue
        return label, ids, True
    return None, [], False


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

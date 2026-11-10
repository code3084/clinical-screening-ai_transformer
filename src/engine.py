"""Model engine used by the app: lazy loading, per-criterion inference, live decoding benchmark.
Uses OUR decoders (src/decoding.py): pure greedy argmax, no repetition penalty."""
import gc
import threading

import torch

from . import config
from .common import parse_output
from .decoding import (get_eos_ids, greedy_kvcache, greedy_nocache, speculative_greedy)
from .model_utils import chat_prompt, load_model, load_tokenizer
from .run_decoding import timed

VARIANTS = {"finetuned": "Fine-tuned LoRA (recommended)", "base": "Base model, zero-shot (for comparison)"}


class Engine:
    def __init__(self, base=None, adapter=None, draft=None, draft_adapter=None, device=None, dtype=None):
        self.base, self.draft_name = base or config.BASE_MODEL, draft or config.DRAFT_MODEL
        self.adapter = adapter or config.default_adapter()
        self.draft_adapter = draft_adapter or config.default_draft_adapter()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self._lock = threading.RLock()
        self._tok, self._targets, self._draft, self.draft_note = None, {}, None, ""

    # ------------------------------------------------------------------ loading
    @property
    def tok(self):
        with self._lock:
            if self._tok is None:
                self._tok = load_tokenizer(self.base)
            return self._tok

    def target(self, variant="finetuned"):
        with self._lock:
            if variant not in self._targets:
                if self.device == "cpu":                      # keep one target resident on small-RAM machines
                    self._targets.clear(); gc.collect()
                ad = self.adapter if variant == "finetuned" else None
                try:
                    self._targets[variant] = load_model(self.base, ad, self.dtype, self.device)
                except Exception as e:
                    if variant == "finetuned":
                        raise RuntimeError(f"could not load LoRA adapter '{self.adapter}' ({type(e).__name__}: {e}). "
                                           "Set ADAPTER_REPO in src/config.py or pass --adapter.") from e
                    raise
            return self._targets[variant]

    def draft_model(self):
        with self._lock:
            if self._draft is None:
                try:
                    self._draft = load_model(self.draft_name, self.draft_adapter, self.dtype, self.device)
                    self.draft_note = f"draft adapter: {self.draft_adapter}"
                except Exception as e:                         # fall back to the un-tuned draft (still valid for speculative decoding)
                    self._draft = load_model(self.draft_name, None, self.dtype, self.device)
                    self.draft_note = f"base draft used (adapter unavailable: {type(e).__name__})"
            return self._draft

    def ensure_ready(self, variant, mode):
        self.tok; self.target(variant)
        if mode == "speculative":
            self.draft_model()

    def info(self):
        d = dict(base=self.base, adapter=self.adapter, draft=self.draft_name, draft_adapter=self.draft_adapter,
                 device=self.device, dtype=str(self.dtype).replace("torch.", ""), loaded=sorted(self._targets))
        m = next(iter(self._targets.values()), None)
        if m is not None:
            c = m.config
            d.update(params_b=sum(p.numel() for p in m.parameters()) / 1e9, layers=c.num_hidden_layers,
                     hidden=c.hidden_size, heads=c.num_attention_heads, kv_heads=c.num_key_value_heads,
                     vocab=c.vocab_size, max_positions=c.max_position_embeddings)
        return d

    # ------------------------------------------------------------------ inference
    def _ids(self, sentences, ctype, ctext):
        text = chat_prompt(self.tok, sentences, ctype, ctext)
        return torch.tensor([self.tok(text, add_special_tokens=False).input_ids], device=self.device)

    def count_tokens(self, sentences, ctype, ctext):
        return len(self.tok(chat_prompt(self.tok, sentences, ctype, ctext), add_special_tokens=False).input_ids)

    def _run(self, model, ids, mode, k, max_new, eos):
        if mode == "nocache":
            return greedy_nocache(model, ids, max_new, eos)
        if mode == "speculative":
            return speculative_greedy(model, self.draft_model(), ids, max_new, k=k, eos_ids=eos)
        return greedy_kvcache(model, ids, max_new, eos)

    def predict(self, sentences, ctype, ctext, variant="finetuned", mode="kvcache", k=4, max_new=64):
        with self._lock:
            model = self.target(variant)
            ids = self._ids(sentences, ctype, ctext)
            toks, st = self._run(model, ids, mode, k, max_new, get_eos_ids(self.tok, model))
        raw = self.tok.decode(toks, skip_special_tokens=True)
        label, ev, ok = parse_output(raw, ctype, len(sentences))
        return dict(label=label, evidence=ev, valid=ok, raw=raw, stats=st, prompt_tokens=ids.shape[1])

    def bench(self, sentences, ctype, ctext, variant="finetuned", k=4, reps=3, fixed_len=None, new_tokens=64):
        """Run all three decoding paths on the SAME prompt; returns one row per path."""
        with self._lock:
            model, draft = self.target(variant), self.draft_model()
            ids, eos = self._ids(sentences, ctype, ctext), get_eos_ids(self.tok, model)
            if fixed_len:                                      # fixed-length mode: tile the prompt, ignore EOS
                ids = ids.repeat(1, -(-fixed_len // ids.shape[1]))[:, :fixed_len]
                eos = None
            paths = {"No cache (reference)": lambda: greedy_nocache(model, ids, new_tokens, eos),
                     "KV cache": lambda: greedy_kvcache(model, ids, new_tokens, eos),
                     f"Speculative (k={k})": lambda: speculative_greedy(model, draft, ids, new_tokens, k=k, eos_ids=eos)}
            rows, ref = [], None
            for name, fn in paths.items():
                toks, st = timed(fn, warm=1, reps=reps)
                ref = toks if ref is None else ref
                rows.append(dict(mode=name, prompt_len=ids.shape[1], new_tokens=len(toks), equal_to_reference=toks == ref,
                                 prefill_s=st["prefill_s"], decode_s=st["decode_s"], total_s=st["total_s"],
                                 decode_tok_per_s=st["decode_tok_per_s"], e2e_tok_per_s=st["e2e_tok_per_s"],
                                 peak_extra_mb=st.get("peak_extra_mb"), acceptance_rate=st.get("acceptance_rate")))
            return rows

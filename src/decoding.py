"""Tasks 3 & 5: explicit KV-cached greedy decoding and greedy speculative decoding.

Every decoder drives the model with explicit forward passes and manages itself:
  * cache reuse/update   (past_key_values object, its length asserted after each step)
  * position handling    (position_ids = absolute index of each fed token)
  * masking              (attention_mask covers cached + new tokens; HF builds the
                          causal mask from it and from the cache positions)
All decoders are batch-size 1 and deterministic (argmax). `eos_ids=None` disables
early stopping so exact output lengths can be benchmarked.
"""
import time

import torch

# ----------------------------------------------------------------------------- helpers


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_eos_ids(tok, model):
    ids = {tok.eos_token_id}
    g = getattr(model, "generation_config", None)
    e = getattr(g, "eos_token_id", None)
    if e is not None:
        ids |= set(e if isinstance(e, (list, tuple)) else [e])
    return ids


def encode_prompt(tok, rec, device):
    from .model_utils import chat_prompt
    text = chat_prompt(tok, rec["sentences"], rec["criterion_type"], rec["criterion_text"])
    return torch.tensor([tok(text, add_special_tokens=False).input_ids], device=device)


def make_prompt_ids(tok, recs, target_len, device):
    """Natural-looking prompt of EXACTLY target_len tokens: concatenated real chat prompts, cut."""
    from .model_utils import chat_prompt
    ids = []
    for r in recs:
        ids += tok(chat_prompt(tok, r["sentences"], r["criterion_type"], r["criterion_text"]),
                   add_special_tokens=False).input_ids
        if len(ids) >= target_len:
            break
    assert len(ids) >= target_len, "not enough text to build the requested prompt length"
    return torch.tensor([ids[:target_len]], device=device)


def kv_bytes_per_token(config, dtype=torch.float16):
    """2 (K and V) x layers x kv_heads x head_dim x bytes. Grows linearly with sequence length."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * torch.tensor([], dtype=dtype).element_size()


def cache_len(cache):
    return cache.get_seq_length()


def crop_cache(cache, n):
    """Roll the cache back to its first n positions (needed to discard rejected drafts).

    Cache.crop() semantics differ across transformers versions (4.x: absolute length,
    5.x: negative = tokens to remove). A NEGATIVE argument means "remove that many
    tokens" in both, so we always compute the number of positions to drop.
    """
    remove = cache_len(cache) - n
    if remove <= 0:
        return
    if hasattr(cache, "crop"):
        cache.crop(-remove)
    elif hasattr(cache, "layers"):                     # layered caches without crop()
        for layer in cache.layers:
            layer.keys, layer.values = layer.keys[..., :n, :], layer.values[..., :n, :]
    else:                                              # legacy lists
        for i in range(len(cache.key_cache)):
            cache.key_cache[i], cache.value_cache[i] = cache.key_cache[i][..., :n, :], cache.value_cache[i][..., :n, :]
    assert cache_len(cache) == n, f"crop failed: {cache_len(cache)} != {n}"


def forward(model, ids, *, mask, pos, cache=None, use_cache=True, keep=1):
    """One explicit forward pass; returns (logits for the last `keep` positions, cache)."""
    kw = dict(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=use_cache)
    for name in ("logits_to_keep", "num_logits_to_keep"):     # skip the full-vocab matmul on old positions
        try:
            out = model(**kw, **{name: keep})
            return out.logits[:, -keep:], out.past_key_values
        except TypeError:
            continue
    out = model(**kw)
    return out.logits[:, -keep:], out.past_key_values


def _ones(n, dev):
    return torch.ones(1, n, dtype=torch.long, device=dev)


def _arange(a, b, dev):
    return torch.arange(a, b, device=dev)[None]


def _stats(t0, t1, t2, n_new, extra=None):
    d = dict(prefill_s=t1 - t0, decode_s=t2 - t1, total_s=t2 - t0, new_tokens=n_new,
             decode_tok_per_s=(n_new - 1) / (t2 - t1) if n_new > 1 and t2 > t1 else 0.0,
             e2e_tok_per_s=n_new / (t2 - t0))
    if torch.cuda.is_available():
        d["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    d.update(extra or {})
    return d

# ----------------------------------------------------------------------------- reference: no cache


@torch.no_grad()
def greedy_nocache(model, prompt_ids, max_new, eos_ids=None):
    """Reference path: every step re-feeds the WHOLE sequence (O(n^2) total work)."""
    dev, ids, new = prompt_ids.device, prompt_ids, []
    sync(); t0 = time.perf_counter(); t1 = t0
    for step in range(max_new):
        L = ids.shape[1]
        logits, _ = forward(model, ids, mask=_ones(L, dev), pos=_arange(0, L, dev), cache=None, use_cache=False, keep=1)
        nxt = int(logits[0, -1].argmax())
        if step == 0:
            sync(); t1 = time.perf_counter()
        new.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], 1)
        if eos_ids and nxt in eos_ids:
            break
    sync(); t2 = time.perf_counter()
    return new, _stats(t0, t1, t2, len(new))

# ----------------------------------------------------------------------------- KV-cached


@torch.no_grad()
def greedy_kvcache(model, prompt_ids, max_new, eos_ids=None):
    """Prefill once (fills the cache), then feed ONE token per step reusing/extending the cache."""
    dev, P = prompt_ids.device, prompt_ids.shape[1]
    sync(); t0 = time.perf_counter()
    logits, cache = forward(model, prompt_ids, mask=_ones(P, dev), pos=_arange(0, P, dev), cache=None, keep=1)
    assert cache_len(cache) == P
    new = [int(logits[0, -1].argmax())]
    sync(); t1 = time.perf_counter()
    while len(new) < max_new and not (eos_ids and new[-1] in eos_ids):
        past = cache_len(cache)                           # P + tokens already consumed
        logits, cache = forward(model, torch.tensor([[new[-1]]], device=dev),
                                mask=_ones(past + 1, dev),   # cached + the new token
                                pos=torch.tensor([[past]], device=dev),   # absolute position of the new token
                                cache=cache, keep=1)
        assert cache_len(cache) == past + 1               # cache grew by exactly one position
        new.append(int(logits[0, -1].argmax()))
    sync(); t2 = time.perf_counter()
    return new, _stats(t0, t1, t2, len(new), dict(cache_len_end=cache_len(cache)))

# ----------------------------------------------------------------------------- speculative


@torch.no_grad()
def speculative_greedy(target, draft, prompt_ids, max_new, k=4, eos_ids=None):
    """Greedy speculative decoding (draft proposes k tokens, target verifies them in ONE pass).

    Invariants at the start of every round (seq = prompt + accepted tokens, L = len(seq)):
      * seq[-1] is 'pending': known/accepted but NOT yet in either cache.
      * target cache covers exactly L-1 positions; draft cache covers <= L-1 positions.
    Round: draft feeds seq[len(Dc):] then k-1 own tokens -> d_1..d_k.
           target feeds [pending, d_1..d_k] -> greedy predictions g_0..g_k, where g_i is the
           prediction after the i-th fed token. Accept the longest prefix with d_i == g_{i-1};
           the next token is g_m (correction if m<k, 'bonus' if m==k). Output is therefore
           token-for-token what target-only greedy decoding produces.
    Sync: target cache is cropped to L+m, draft cache to min(len, L+m) (rejected drafts removed).
    """
    dev, P = prompt_ids.device, prompt_ids.shape[1]
    st = dict(rounds=0, drafted=0, accepted=0, target_forwards=0, draft_forwards=0, accept_hist=[0] * (k + 1))
    sync(); t0 = time.perf_counter()
    lt, Tc = forward(target, prompt_ids, mask=_ones(P, dev), pos=_arange(0, P, dev), keep=1)
    _, Dc = forward(draft, prompt_ids, mask=_ones(P, dev), pos=_arange(0, P, dev), keep=1)
    st["target_forwards"] += 1; st["draft_forwards"] += 1
    seq = prompt_ids[0].tolist() + [int(lt[0, -1].argmax())]
    n_new = 1
    sync(); t1 = time.perf_counter()
    done = bool(eos_ids and seq[-1] in eos_ids)

    while n_new < max_new and not done:
        L = len(seq)
        k_eff = min(k, max_new - n_new - 1) if max_new - n_new > 1 else 0   # never overshoot max_new
        drafts = []
        if k_eff > 0:                                                         # --- draft phase
            d_len = cache_len(Dc)
            chunk = torch.tensor([seq[d_len:]], device=dev)                   # all tokens the draft hasn't seen
            lg, Dc = forward(draft, chunk, mask=_ones(L, dev), pos=_arange(d_len, L, dev), cache=Dc, keep=1)
            drafts.append(int(lg[0, -1].argmax())); st["draft_forwards"] += 1
            for _ in range(k_eff - 1):
                past = cache_len(Dc)
                lg, Dc = forward(draft, torch.tensor([[drafts[-1]]], device=dev), mask=_ones(past + 1, dev),
                                 pos=torch.tensor([[past]], device=dev), cache=Dc, keep=1)
                drafts.append(int(lg[0, -1].argmax())); st["draft_forwards"] += 1
        # --- verify phase: one target pass over [pending, d_1..d_k]
        t_len = cache_len(Tc)
        assert t_len == L - 1
        chunk_ids = [seq[-1]] + drafts
        lg, Tc = forward(target, torch.tensor([chunk_ids], device=dev), mask=_ones(t_len + len(chunk_ids), dev),
                         pos=_arange(t_len, t_len + len(chunk_ids), dev), cache=Tc, keep=len(chunk_ids))
        st["target_forwards"] += 1
        g = lg[0].argmax(-1).tolist()
        m = 0
        while m < k_eff and drafts[m] == g[m]:
            m += 1
        new_tokens = drafts[:m] + [g[m]]
        st["rounds"] += 1; st["drafted"] += k_eff; st["accepted"] += m; st["accept_hist"][m] += 1
        # --- cache synchronisation: keep only positions whose tokens are in the new sequence (minus pending)
        crop_cache(Tc, L + m)
        if k_eff > 0:
            crop_cache(Dc, min(cache_len(Dc), L + m))
        if eos_ids:
            for i, t in enumerate(new_tokens):
                if t in eos_ids:
                    new_tokens, done = new_tokens[:i + 1], True
                    break
        seq += new_tokens
        n_new += len(new_tokens)
    sync(); t2 = time.perf_counter()
    gen = seq[P:]
    extra = dict(st, acceptance_rate=(st["accepted"] / st["drafted"]) if st["drafted"] else 0.0,
                 tokens_per_target_forward=n_new / st["target_forwards"], k=k)
    return gen, _stats(t0, t1, t2, len(gen), extra)

# ----------------------------------------------------------------------------- library reference


@torch.no_grad()
def hf_greedy(model, prompt_ids, max_new, eos_ids=None, assistant=None):
    """Library reference (generate / assisted generation) used ONLY to cross-check our decoders."""
    kw = dict(max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
              repetition_penalty=1.0,          # Qwen's generation_config ships a penalty -> not pure greedy
              attention_mask=_ones(prompt_ids.shape[1], prompt_ids.device),
              eos_token_id=sorted(eos_ids) if eos_ids else None, pad_token_id=0)
    if not eos_ids:
        kw["min_new_tokens"] = max_new          # force exact length (disables early stop)
        kw["eos_token_id"] = None
    if assistant is not None:
        kw["assistant_model"] = assistant
    sync(); t0 = time.perf_counter()
    out = model.generate(prompt_ids, **kw)
    sync(); t1 = time.perf_counter()
    new = out[0, prompt_ids.shape[1]:].tolist()
    return new, dict(total_s=t1 - t0, new_tokens=len(new))

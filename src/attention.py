"""Task 4: reference and tiled (online-softmax) scaled dot-product attention.

Shapes: q [B,H,Lq,D], k [B,H,Lk,D], v [B,H,Lk,Dv].
Causal convention (bottom-right aligned, as during cached decoding): query i has
absolute position (Lk-Lq)+i and may attend keys j <= (Lk-Lq)+i. Requires Lq <= Lk.
Internals are computed in at least float32 and cast back to the input dtype.
"""
import math

import torch


def _acc_dtype(t):
    return torch.promote_types(t.dtype, torch.float32)


def _check(q, k, causal):
    if causal and q.shape[-2] > k.shape[-2]:
        raise ValueError("causal attention needs Lq <= Lk (otherwise some rows are fully masked)")


def reference_attention(q, k, v, causal=True, scale=None):
    """Plain attention: materialises the full [Lq, Lk] score matrix (O(L^2) memory)."""
    _check(q, k, causal)
    acc = _acc_dtype(q)
    Lq, Lk = q.shape[-2], k.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = (q.to(acc) @ k.to(acc).transpose(-1, -2)) * scale
    if causal:
        qi = torch.arange(Lq, device=q.device)[:, None] + (Lk - Lq)
        kj = torch.arange(Lk, device=q.device)[None, :]
        s = s.masked_fill(kj > qi, float("-inf"))
    s = s - s.amax(dim=-1, keepdim=True)          # stable softmax: subtract row max
    p = torch.exp(s)
    p = p / p.sum(dim=-1, keepdim=True)
    return (p @ v.to(acc)).to(q.dtype)


def tiled_attention(q, k, v, causal=True, block_q=64, block_k=64, scale=None):
    """Blocked attention with ONLINE softmax (FlashAttention-style recurrence).

    For each query block keep running row-max m, normaliser l and unnormalised
    output o. For each key block with scores s:
        m' = max(m, rowmax(s));  p = exp(s - m');  a = exp(m - m')
        l  = l*a + rowsum(p);    o = o*a + p @ V;  m = m'
    and finally out = o / l. Only a [block_q, block_k] tile of scores exists at a
    time, so extra memory is O(block) instead of O(Lq*Lk). Key blocks lying wholly
    in the causal future are skipped; only diagonal blocks get an explicit mask.
    """
    _check(q, k, causal)
    acc = _acc_dtype(q)
    B, H, Lq, _ = q.shape
    Lk, Dv = k.shape[-2], v.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    off = Lk - Lq
    out = torch.empty(B, H, Lq, Dv, dtype=acc, device=q.device)
    for qs in range(0, Lq, block_q):
        qe = min(qs + block_q, Lq)
        qb = q[..., qs:qe, :].to(acc)
        m = torch.full((B, H, qe - qs), float("-inf"), dtype=acc, device=q.device)
        l = torch.zeros((B, H, qe - qs), dtype=acc, device=q.device)
        o = torch.zeros((B, H, qe - qs, Dv), dtype=acc, device=q.device)
        for ks in range(0, Lk, block_k):
            ke = min(ks + block_k, Lk)
            if causal and ks > (qe - 1) + off:     # whole block is in the future
                break
            s = (qb @ k[..., ks:ke, :].to(acc).transpose(-1, -2)) * scale
            if causal and ke - 1 > qs + off:       # block crosses the diagonal -> mask
                qi = torch.arange(qs, qe, device=q.device)[:, None] + off
                kj = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1))
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            p = torch.exp(s - m_safe[..., None])   # exp(-inf)=0 for masked entries
            alpha = torch.exp(m - m_safe)          # rescales what was accumulated so far
            l = l * alpha + p.sum(dim=-1)
            o = o * alpha[..., None] + p @ v[..., ks:ke, :].to(acc)
            m = m_new
        out[..., qs:qe, :] = o / l[..., None]
    return out.to(q.dtype)

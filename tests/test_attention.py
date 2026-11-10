import pytest
import torch

from src.attention import reference_attention, tiled_attention

SHAPES = [(1, 2, 16, 16, 8), (2, 3, 50, 50, 16), (1, 2, 37, 101, 32), (1, 1, 1, 129, 64), (2, 2, 130, 130, 24)]


def naive(q, k, v, causal):                      # independent float64 implementation (torch.tril mask)
    s = q.double() @ k.double().transpose(-1, -2) / q.shape[-1] ** 0.5
    if causal:
        s = s.masked_fill(~torch.tril(torch.ones(q.shape[2], k.shape[2], dtype=torch.bool), diagonal=k.shape[2] - q.shape[2]), float("-inf"))
    return torch.softmax(s, -1) @ v.double()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("blocks", [(16, 16), (64, 32), (7, 13)])
def test_tiled_matches_reference(shape, causal, blocks):
    B, H, Lq, Lk, D = shape
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, L, D, dtype=torch.float64) for L in (Lq, Lk, Lk))
    truth = naive(q, k, v, causal)
    assert (reference_attention(q, k, v, causal) - truth).abs().max() < 1e-12
    assert (tiled_attention(q, k, v, causal, *blocks) - truth).abs().max() < 1e-12
    out32 = tiled_attention(q.float(), k.float(), v.float(), causal, *blocks)
    assert (out32.double() - truth).abs().max() < 1e-5


def test_large_logits_are_stable():
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 2, 64, 16) * 40 for _ in range(3))      # huge scores would overflow a naive exp
    out = tiled_attention(q, k, v, True, 16, 16)
    assert torch.isfinite(out).all()
    assert (out - reference_attention(q, k, v, True)).abs().max() < 1e-4


def test_causality_no_future_leak():
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 1, 32, 8) for _ in range(3))
    base = tiled_attention(q, k, v, True, 8, 8)
    v2 = v.clone(); v2[..., 20:, :] += 100.0                          # change only future positions
    assert torch.allclose(base[..., :20, :], tiled_attention(q, k, v2, True, 8, 8)[..., :20, :])


def test_lq_greater_than_lk_rejected():
    q, k = torch.randn(1, 1, 8, 4), torch.randn(1, 1, 4, 4)
    with pytest.raises(ValueError):
        tiled_attention(q, k, k, causal=True)

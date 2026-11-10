"""Decoder-logic tests on TINY RANDOM Qwen2 models (CPU, fp32): fast, deterministic, no downloads."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.decoding import (cache_len, crop_cache, forward, greedy_kvcache, greedy_nocache, hf_greedy,
                          speculative_greedy)


def tiny(seed, layers):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=120, hidden_size=48, intermediate_size=96, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=False)
    m = Qwen2ForCausalLM(cfg).eval()
    m.lm_head.weight.data *= 30                      # sharpen logits -> clear argmax margins
    return m


@pytest.fixture(scope="module")
def target():
    return tiny(0, 3)


@pytest.fixture(scope="module")
def draft():
    return tiny(1, 1)


def prompt(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 120, (1, n), generator=g)


@pytest.mark.parametrize("P,N", [(5, 20), (40, 30), (100, 12)])
def test_kvcache_matches_nocache_and_hf(target, P, N):
    ids = prompt(P, P)
    ref, _ = greedy_nocache(target, ids, N)
    kv, st = greedy_kvcache(target, ids, N)
    assert kv == ref and st["cache_len_end"] == P + N - 1
    assert hf_greedy(target, ids, N)[0] == ref


@pytest.mark.parametrize("k", [1, 2, 4, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_speculative_equals_greedy(target, draft, k, seed):
    ids = prompt(17 + seed, seed)
    ref, _ = greedy_nocache(target, ids, 40)
    sp, st = speculative_greedy(target, draft, ids, 40, k=k)
    assert sp == ref and len(sp) == 40
    assert st["accepted"] <= st["drafted"]


def test_speculative_full_acceptance_with_identical_draft(target):
    ids = prompt(20, 5)
    ref, _ = greedy_nocache(target, ids, 33)
    sp, st = speculative_greedy(target, target, ids, 33, k=4)      # draft == target -> every draft accepted
    assert sp == ref and st["acceptance_rate"] == 1.0
    assert st["tokens_per_target_forward"] > 3


def test_hf_assisted_agrees(target, draft):
    ids = prompt(25, 3)
    ref, _ = greedy_nocache(target, ids, 30)
    assert hf_greedy(target, ids, 30, assistant=draft)[0] == ref


def test_eos_stops_identically(target, draft):
    ids = prompt(15, 9)
    free, _ = greedy_nocache(target, ids, 40)
    eos = {free[7]}                                                # token the model really emits -> early stop
    ref, _ = greedy_nocache(target, ids, 40, eos)
    assert len(ref) <= 8 and ref[-1] in eos
    assert greedy_kvcache(target, ids, 40, eos)[0] == ref
    assert speculative_greedy(target, draft, ids, 40, k=4, eos_ids=eos)[0] == ref
    assert speculative_greedy(target, target, ids, 40, k=4, eos_ids=eos)[0] == ref


@pytest.mark.parametrize("N", [1, 2, 3])
def test_tiny_max_new(target, draft, N):
    ids = prompt(10, 4)
    ref, _ = greedy_nocache(target, ids, N)
    assert speculative_greedy(target, draft, ids, N, k=4)[0] == ref


def test_crop_cache_restores_state(target):
    ids = prompt(30, 6)
    L = ids.shape[1]
    _, cache = forward(target, ids, mask=torch.ones(1, L, dtype=torch.long), pos=torch.arange(L)[None])
    crop_cache(cache, 20)
    assert cache_len(cache) == 20
    lg_c, _ = forward(target, ids[:, 20:], mask=torch.ones(1, L, dtype=torch.long), pos=torch.arange(20, L)[None],
                      cache=cache, keep=10)
    lg_f, _ = forward(target, ids, mask=torch.ones(1, L, dtype=torch.long), pos=torch.arange(L)[None], use_cache=False, keep=10)
    assert (lg_c - lg_f).abs().max() < 1e-4

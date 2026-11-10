"""Model/tokenizer loading and chat-prompt construction (shared by train, eval,
KV-cache, speculative decoding and the app)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import build_user_prompt

SYSTEM = "You are a careful clinical trial eligibility screening assistant. Reply with JSON only."
DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def load_tokenizer(name=DEFAULT_BASE):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"          # required for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(name=DEFAULT_BASE, dtype=torch.float16, device="cuda"):
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:                   # older transformers
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return model.to(device)


def load_model(name=DEFAULT_BASE, adapter=None, dtype=torch.float16, device="cuda"):
    model = load_base_model(name, dtype, device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()   # plain model -> same speed as base, simpler cache code
    return model.eval()


def chat_prompt(tok, sentences, criterion_type, criterion_text):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_prompt(sentences, criterion_type, criterion_text)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

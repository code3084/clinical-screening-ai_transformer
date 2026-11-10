"""Single place to configure model locations. 

Resolution order for each adapter: command-line flag > environment variable >
local folder (if it exists) > the Hugging Face repo id below.
"""
import os

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_REPO = "code3084/clinicalScreen-qwen2.5-1.5b-lora"              # fine-tuned target LoRA adapter (public)
DRAFT_ADAPTER_REPO = "code3084/clinicalScreen_draft-qwen2.5-0.5b-lora"  # fine-tuned draft LoRA adapter (public)

LOCAL_ADAPTER = "runs/lora_r16/best"
LOCAL_DRAFT_ADAPTER = "runs/draft_lora/best"


def default_adapter():
    return os.environ.get("TRIALSCREEN_ADAPTER") or (LOCAL_ADAPTER if os.path.isdir(LOCAL_ADAPTER) else ADAPTER_REPO)


def default_draft_adapter():
    return os.environ.get("TRIALSCREEN_DRAFT_ADAPTER") or (
        LOCAL_DRAFT_ADAPTER if os.path.isdir(LOCAL_DRAFT_ADAPTER) else DRAFT_ADAPTER_REPO)

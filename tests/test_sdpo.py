"""Tests for the SDPO trainer using a tiny model.

We use ``hf-internal-testing/tiny-random-LlamaForCausalLM`` because (a) it
ships LoRA-friendly target modules (q_proj, k_proj, …) and (b) it's <10 MB
so the test suite stays fast.
"""

from __future__ import annotations

import importlib.util

import pytest

# Skip the whole module if local extras aren't installed.
pytestmark = pytest.mark.local_model

if not all(
    importlib.util.find_spec(m) for m in ("torch", "transformers", "peft")
):  # pragma: no cover
    pytest.skip("local extras not installed", allow_module_level=True)


import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanoresearch.planner.adapters import AdapterManager, DEFAULT_TARGET_MODULES
from nanoresearch.planner.sdpo import (
    SDPOConfig,
    SDPOExample,
    SDPOTrainer,
    sdpo_loss,
)

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    # Force a permissive template — Llama's default enforces strict alternation
    # which is fine for inference but not what we want in unit tests.
    tok.chat_template = (
        "{% for m in messages %}"
        "<|{{m.role}}|>\n{{m.content}}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )
    return model, tok


@pytest.fixture
def lora_model(tiny_model_and_tokenizer):
    model, _ = tiny_model_and_tokenizer
    mgr = AdapterManager()
    # tiny-random-Llama has q/k/v/o_proj; gate/up/down may not all exist, so
    # we narrow the targets to ones we know are present in the architecture.
    cfg = mgr.make_lora_config(target_modules=("q_proj", "v_proj"))
    peft = get_peft_model(model, cfg, adapter_name="alice")
    return peft


def test_sdpo_loss_returns_scalar_with_gradient(tiny_model_and_tokenizer, lora_model):
    _, tok = tiny_model_and_tokenizer
    ex = SDPOExample(
        prompt_messages=[
            {"role": "system", "content": "You plan experiments."},
            {"role": "user", "content": "Plan a tiny MLP run on MNIST."},
        ],
        response="step1: load MNIST. step2: train 2-layer MLP. step3: eval.",
        feedback="Be more concise and use bullet points.",
    )
    loss = sdpo_loss(
        model=lora_model,
        tokenizer=tok,
        example=ex,
        config=SDPOConfig(max_steps=1, max_trained_tokens=64),
        device="cpu",
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    # At least one LoRA param must have received a gradient.
    grads_present = any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in lora_model.parameters()
        if p.requires_grad
    )
    assert grads_present, "no LoRA gradients flowed — SDPO connectivity is broken"


def test_sdpo_trainer_reduces_loss_on_repeated_example(
    tiny_model_and_tokenizer, lora_model, tmp_path, monkeypatch
):
    """A noisy but reliable signal: re-fitting the same example should drive
    mean loss down over enough steps."""
    monkeypatch.setenv("LORA_ADAPTERS_DIR", str(tmp_path))
    _, tok = tiny_model_and_tokenizer
    ex = SDPOExample(
        prompt_messages=[
            {"role": "user", "content": "What is the plan?"},
        ],
        response="run baselines first then evaluate the proposed method",
        feedback="Prefer JSON-style output with explicit fields.",
    )
    mgr = AdapterManager(root=tmp_path)
    cfg = SDPOConfig(
        learning_rate=5e-3,
        max_steps=8,
        max_trained_tokens=32,
        gradient_accumulation_steps=1,
    )
    trainer = SDPOTrainer(
        lora_model, tok, user_id="alice", device="cpu", config=cfg, adapter_manager=mgr
    )
    summary = trainer.train([ex])
    assert summary["num_steps"] == 8
    # Mean across the run should be below the first step's loss — a more
    # reliable monotonicity signal than a strict last < first comparison.
    assert summary["mean_loss"] < summary["first_loss"], summary


def test_sdpo_trainer_saves_and_reloads(tmp_path, tiny_model_and_tokenizer):
    _, tok = tiny_model_and_tokenizer
    # Fresh PEFT wrap so paths don't collide with other tests
    model = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
    mgr = AdapterManager(root=tmp_path)
    cfg = mgr.make_lora_config(target_modules=("q_proj", "v_proj"))
    peft = get_peft_model(model, cfg, adapter_name="bob")

    ex = SDPOExample(
        prompt_messages=[{"role": "user", "content": "Plan"}],
        response="step1: do X",
        feedback="be terse",
    )
    trainer = SDPOTrainer(
        peft, tok, user_id="bob", device="cpu",
        config=SDPOConfig(max_steps=2, max_trained_tokens=16),
        adapter_manager=mgr,
    )
    trainer.train([ex])
    path = trainer.save()
    assert (path / "adapter_config.json").exists()
    assert mgr.exists("bob")
    assert "bob" in mgr.list_users()

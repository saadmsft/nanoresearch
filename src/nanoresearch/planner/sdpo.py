"""Self-Distillation Policy Optimization (SDPO).

Implements paper §3.3.2, Eq. 14-15. Free-form user feedback ``ℱ`` is converted
into a dense, token-level learning signal that updates only the user's LoRA
adapter — no reward model, no preference annotations.

Given:
- Orchestrator input ``x``
- Planner's initial trajectory ``y ~ π_θ(·|x)``
- Free-form feedback ``ℱ``

The student is ``π_θ(·|x, y_<t)`` and the self-teacher is
``π_θ(·|x, ℱ, y_<t)``. The SDPO gradient (Eq. 14) is a logit-level policy
gradient with the per-token advantage (Eq. 15)::

    A_t = log π_θ(ŷ_t | x, ℱ, y_<t)  -  log π_θ(ŷ_t | x, y_<t)

where ``ŷ_t ~ π_θ(·|x, y_<t)``.

The loss we minimise (with sign flipped to gradient-descent) is::

    L(θ) = - E_y [ Σ_t  stopgrad(A_t)  · log π_θ(ŷ_t | x, y_<t) ]

LoRA params are the only trainable variables; base weights stay frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import get_settings
from ..logging import get_logger
from .adapters import AdapterManager

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

_log = get_logger(__name__)


@dataclass
class SDPOConfig:
    """Hyper-parameters for one SDPO update round.

    Defaults are tuned for Qwen2.5-7B + LoRA on M1 Max (32 GB).
    """

    learning_rate: float = 1e-4
    max_steps: int = 50
    max_sequence_length: int = 2048
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.0
    advantage_clip: float = 5.0  # clamp |A_t| to bound updates
    seed: int = 42
    # Train only the most recent N tokens of y (the trajectory) — saves memory
    # while keeping the feedback signal dense over the generated content.
    max_trained_tokens: int = 512


@dataclass
class SDPOExample:
    """One (x, y, F) triple — i.e. one stage's feedback round."""

    prompt_messages: list[dict[str, str]]
    """The chat messages forming ``x`` (system + user)."""
    response: str
    """The planner's initial trajectory ``y``."""
    feedback: str
    """Free-form user feedback ``ℱ``."""


def sdpo_loss(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    example: SDPOExample,
    config: SDPOConfig,
    device: str,
) -> torch.Tensor:
    """Compute the SDPO loss for a single example.

    Implements Eq. 14-15. Returns a scalar :class:`torch.Tensor` with a
    gradient w.r.t. the LoRA params (because base weights are frozen).

    The two forward passes are:
    1. ``student``  = π_θ(·|x, y_<t)      — gradient-tracked
    2. ``teacher``  = π_θ(·|x, ℱ, y_<t)   — no_grad (stop-grad on A_t)
    """
    import torch

    # ---- 1. Build the two prompts -------------------------------------
    # student_prompt:    [chat history]                              -> y
    # teacher_prompt:    [chat history with feedback merged in]      -> y
    #
    # We append feedback to the *last user message* rather than adding a new
    # user turn so the construction is compatible with chat templates that
    # enforce strict user/assistant alternation. This matches the paper's
    # framing — feedback ℱ is part of the orchestrator's input x to the
    # teacher distribution π_θ(·|x, ℱ, y_<t).
    student_messages = list(example.prompt_messages)

    teacher_messages = [dict(m) for m in example.prompt_messages]
    feedback_block = (
        "\n\n[USER FEEDBACK ON PRIOR ANSWER]\n"
        f"{example.feedback}\n"
        "[END FEEDBACK]\n"
        "Internalise this feedback when answering."
    )
    if teacher_messages and teacher_messages[-1]["role"] == "user":
        teacher_messages[-1]["content"] = teacher_messages[-1]["content"] + feedback_block
    else:
        teacher_messages.append({"role": "user", "content": feedback_block.strip()})

    student_prompt_text: str = tokenizer.apply_chat_template(
        student_messages, tokenize=False, add_generation_prompt=True
    )
    teacher_prompt_text: str = tokenizer.apply_chat_template(
        teacher_messages, tokenize=False, add_generation_prompt=True
    )

    # ---- 2. Tokenise prompts + response -------------------------------
    student_prompt_ids = tokenizer(
        student_prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    teacher_prompt_ids = tokenizer(
        teacher_prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    response_ids = tokenizer(
        example.response, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    # Cap the response length so we don't OOM
    if response_ids.size(1) > config.max_trained_tokens:
        response_ids = response_ids[:, : config.max_trained_tokens]

    student_input = torch.cat([student_prompt_ids, response_ids], dim=1)
    teacher_input = torch.cat([teacher_prompt_ids, response_ids], dim=1)

    # Truncate from the left if either exceeds max sequence length.
    if student_input.size(1) > config.max_sequence_length:
        student_input = student_input[:, -config.max_sequence_length :]
    if teacher_input.size(1) > config.max_sequence_length:
        teacher_input = teacher_input[:, -config.max_sequence_length :]

    response_len = response_ids.size(1)

    # ---- 3. Forward passes --------------------------------------------
    # Student: gradient-tracked, gives ``log π_θ(ŷ_t | x, y_<t)`` directly.
    student_out = model(student_input)
    student_logits = student_out.logits  # (1, T_s, V)

    # The next-token at position i in the input is at position i+1 in input.
    # We want log-probs of `response_ids[t]` given `prompt + response[:t]`.
    # For a causal LM, logits[:, i, :] predicts input[:, i+1]. So if response
    # tokens occupy positions [P_s, P_s + R), their logits live in
    # [P_s - 1, P_s + R - 1) of the logits tensor.
    student_prompt_len = student_input.size(1) - response_len
    student_logits_slice = student_logits[
        :, student_prompt_len - 1 : student_prompt_len - 1 + response_len, :
    ]
    student_logp = torch.log_softmax(student_logits_slice, dim=-1)

    # Teacher: stop-grad — we only need its log-probs as a target.
    with torch.no_grad():
        teacher_out = model(teacher_input)
        teacher_logits = teacher_out.logits
        teacher_prompt_len = teacher_input.size(1) - response_len
        teacher_logits_slice = teacher_logits[
            :, teacher_prompt_len - 1 : teacher_prompt_len - 1 + response_len, :
        ]
        teacher_logp = torch.log_softmax(teacher_logits_slice, dim=-1)

    # Gather log-probs of the actual response tokens.
    response_ids_for_gather = response_ids[:, :, None]  # (1, R, 1)
    student_logp_y = student_logp.gather(-1, response_ids_for_gather).squeeze(-1)  # (1, R)
    teacher_logp_y = teacher_logp.gather(-1, response_ids_for_gather).squeeze(-1)  # (1, R)

    # ---- 4. Per-token advantage (Eq. 15) ------------------------------
    # A_t = log π_θ(ŷ_t | x, ℱ, y_<t)  -  log π_θ(ŷ_t | x, y_<t)
    # Stop gradient on A_t — only the student log-prob carries grad.
    advantage = (teacher_logp_y - student_logp_y).detach()
    if config.advantage_clip > 0:
        advantage = advantage.clamp(-config.advantage_clip, config.advantage_clip)

    # ---- 5. Policy-gradient loss (Eq. 14) -----------------------------
    # L = - mean(A_t * log π_θ(ŷ_t | x, y_<t))
    loss = -(advantage * student_logp_y).mean()
    return loss


class SDPOTrainer:
    """Drives N SDPO steps for one user adapter.

    Typical usage::

        trainer = SDPOTrainer(model, tokenizer, user_id="alice", device="mps")
        trainer.train([example1, example2, ...])
        trainer.save()
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        user_id: str,
        device: str | None = None,
        config: SDPOConfig | None = None,
        adapter_manager: AdapterManager | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.user_id = user_id
        self.device = device or get_settings().local_model_device.value
        self.config = config or SDPOConfig()
        self.adapter_manager = adapter_manager or AdapterManager()

    def train(self, examples: list[SDPOExample]) -> dict[str, float]:
        """Run SDPO updates over ``examples`` and return summary metrics."""
        import torch
        from torch.optim import AdamW

        # Only LoRA params should have requires_grad=True (set by PEFT).
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                "No trainable parameters — did you attach a LoRA adapter before training?"
            )

        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        torch.manual_seed(self.config.seed)

        self.model.train()
        losses: list[float] = []
        steps = 0
        accum = 0
        optimizer.zero_grad()

        for step in range(self.config.max_steps):
            example = examples[step % len(examples)]
            loss = sdpo_loss(
                model=self.model,
                tokenizer=self.tokenizer,
                example=example,
                config=self.config,
                device=self.device,
            ) / self.config.gradient_accumulation_steps
            loss.backward()
            accum += 1
            losses.append(float(loss.detach().cpu()) * self.config.gradient_accumulation_steps)

            if accum % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                steps += 1

            _log.debug("sdpo_step", step=step, loss=losses[-1])

        # Flush remaining gradients.
        if accum % self.config.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        self.model.eval()

        summary = {
            "first_loss": losses[0] if losses else 0.0,
            "last_loss": losses[-1] if losses else 0.0,
            "mean_loss": sum(losses) / max(len(losses), 1),
            "num_steps": float(self.config.max_steps),
            "num_examples": float(len(examples)),
        }
        _log.info("sdpo_complete", user_id=self.user_id, **summary)
        return summary

    def save(self) -> Path:
        return self.adapter_manager.save(self.model, self.user_id)

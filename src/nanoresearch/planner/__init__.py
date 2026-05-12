"""Planner: local Qwen2.5-7B with per-user LoRA and SDPO-based fine-tuning.

The Planner is the only **trainable** component of NanoResearch. It encodes
each user's implicit preferences directly into a small set of LoRA parameters
via Self-Distillation Policy Optimization (paper §3.3.2, Eq. 14-15).
"""

from .sdpo import SDPOConfig, SDPOTrainer, sdpo_loss
from .planner import Planner, PlannerOutput
from .adapters import AdapterManager

__all__ = [
    "AdapterManager",
    "Planner",
    "PlannerOutput",
    "SDPOConfig",
    "SDPOTrainer",
    "sdpo_loss",
]

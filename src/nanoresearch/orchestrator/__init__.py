"""Orchestrator 𝒪 and stage controllers (paper §3.1, §3.2, §3.3).

The Orchestrator drives the central self-evolution loop::

    retrieve(𝒮, ℳ | C)  →  plan(via π_θ)  →  dispatch(stage agents)
                                           →  reflect over τ
                                           →  update(𝒮, ℳ)

Each stage (Ideation, Planning, Coding, Debug, Analysis, Writing, Review) is
a subclass of :class:`Stage`. Stage-specific logic lands in Phases 4-6; this
module provides the framework that wires everything together.
"""

from .feedback import FeedbackQueue
from .orchestrator import Orchestrator
from .stage import Stage, StageContext, StageResult, StageStatus
from .trajectory import Trajectory, TrajectoryEvent

__all__ = [
    "FeedbackQueue",
    "Orchestrator",
    "Stage",
    "StageContext",
    "StageResult",
    "StageStatus",
    "Trajectory",
    "TrajectoryEvent",
]

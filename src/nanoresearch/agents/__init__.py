"""Stage controllers for the NanoResearch pipeline.

Each module here implements one stage of paper §3.2:

- :mod:`stage1_ideation`   — Ideation phase (literature → hypotheses → h*).
- :mod:`stage1_planning`   — Planning phase (h* → blueprint ℬ + review loop).
- :mod:`stage2_coding`     — Coding + sandboxed execution + debug loop.
- :mod:`stage2_analysis`   — Raw logs → :class:`AnalysisReport`.
- :mod:`stage3_writing`    — Section-by-section LaTeX + reviewer + PDF build.

Stages share :mod:`prompts` for system messages and :mod:`blueprint` /
:mod:`artefacts` for the output schemas.
"""

from .artefacts import (
    AnalysisReport,
    CodeFile,
    CompiledPaper,
    DebugPatch,
    ExecutionResult,
    GeneratedProject,
    PaperCritique,
    PaperDraft,
    PaperSection,
)
from .blueprint import (
    AblationGroup,
    Blueprint,
    BlueprintCritique,
    Hypothesis,
    IdeationArtefacts,
    NoveltyJudgement,
    ProposedMethod,
)
from .stage1_ideation import IdeationStage
from .stage1_planning import PlanningStage
from .stage2_analysis import AnalysisStage
from .stage2_coding import CodingStage
from .stage3_writing import WritingStage

__all__ = [
    "AblationGroup",
    "AnalysisReport",
    "AnalysisStage",
    "Blueprint",
    "BlueprintCritique",
    "CodeFile",
    "CodingStage",
    "CompiledPaper",
    "DebugPatch",
    "ExecutionResult",
    "GeneratedProject",
    "Hypothesis",
    "IdeationArtefacts",
    "IdeationStage",
    "NoveltyJudgement",
    "PaperCritique",
    "PaperDraft",
    "PaperSection",
    "PlanningStage",
    "ProposedMethod",
    "WritingStage",
]

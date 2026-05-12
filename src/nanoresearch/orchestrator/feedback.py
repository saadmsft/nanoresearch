"""Per-stage feedback queue feeding SDPO training rounds.

After each stage end, the user (or simulated scientist) gives free-form
feedback ``ℱ``. We pair it with the orchestrator's input ``x`` and the
planner's initial trajectory ``y`` to make an :class:`SDPOExample`, then
buffer it until the orchestrator decides to flush.

Paper §3.3.2: applied **after each feedback round**, this update
progressively internalises preferences into the planner.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from ..planner.sdpo import SDPOExample


@dataclass
class FeedbackRecord:
    user_id: str
    stage: str
    example: SDPOExample


class FeedbackQueue:
    """Per-user buffer of pending SDPO examples."""

    def __init__(self, max_per_user: int = 64) -> None:
        self._max = max_per_user
        self._buf: dict[str, deque[FeedbackRecord]] = defaultdict(
            lambda: deque(maxlen=self._max)
        )

    def add(
        self,
        *,
        user_id: str,
        stage: str,
        prompt_messages: list[dict[str, str]],
        response: str,
        feedback: str,
    ) -> FeedbackRecord:
        rec = FeedbackRecord(
            user_id=user_id,
            stage=stage,
            example=SDPOExample(
                prompt_messages=prompt_messages,
                response=response,
                feedback=feedback,
            ),
        )
        self._buf[user_id].append(rec)
        return rec

    def pending_for(self, user_id: str) -> list[FeedbackRecord]:
        return list(self._buf.get(user_id, ()))

    def drain(self, user_id: str) -> list[FeedbackRecord]:
        items = list(self._buf.get(user_id, ()))
        if user_id in self._buf:
            self._buf[user_id].clear()
        return items

    def __len__(self) -> int:
        return sum(len(q) for q in self._buf.values())

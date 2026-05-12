"""Natural-language intent parsing for the chat.

The chat sends every user message to ``POST /api/intent`` along with a small
session context. The server asks GPT-5.1 to classify it into one of a fixed
set of actions and to extract any structured fields, then returns a JSON
``Intent`` envelope. The frontend executes the appropriate API call.

This keeps the UI conversational — the user types whatever they want, the
backend figures out what to do.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..llm import AgentRole, ChatMessage, LLMRouter, Role
from ..logging import get_logger

_log = get_logger(__name__)


# ============================================================ DTOs


class IntentSession(BaseModel):
    """Context the chat ships with every intent request."""

    user_id: str | None = None
    run_id: str | None = None
    run_status: str | None = None
    has_profile: bool = False


class IntentRequest(BaseModel):
    text: str
    session: IntentSession


IntentAction = Literal[
    "help",
    "create_user",
    "select_user",
    "update_profile",
    "list_users",
    "start_run",
    "submit_feedback",
    "status",
    "list_skills",
    "list_memories",
    "train_planner",
    "chitchat",
]


class Intent(BaseModel):
    """Server's decision about what to do with the user's message."""

    action: IntentAction
    # Per-action structured fields. All optional so the schema is forgiving.
    user_id: str | None = None
    topic: str | None = None
    feedback: str | None = None
    profile_updates: dict[str, str] = Field(default_factory=dict)
    # A natural reply the chat can show immediately (e.g., for ``chitchat`` or
    # to confirm an action before the side effect lands).
    reply: str = ""


# ============================================================ NLU


_HELP_RE = re.compile(r"^\s*/?(help|\?)\s*$", re.I)
_CREATE_RE = re.compile(r"^/(?:create|newuser)\s+(\S+)\s*$", re.I)
_USER_RE = re.compile(r"^/user\s+(\S+)\s*$", re.I)
_START_RE = re.compile(r"^/(?:start|run)\s+(.{8,})$", re.I)
_SET_RE = re.compile(r"^/set\s+(\w+)\s*=\s*(.+)$", re.I)
_STATUS_RE = re.compile(r"^/status\s*$", re.I)
_SKILLS_RE = re.compile(r"^/skills\s*$", re.I)
_MEMORIES_RE = re.compile(r"^/memories\s*$", re.I)
_TRAIN_RE = re.compile(r"^/(?:train|sdpo)\s*$", re.I)


def parse_local(text: str) -> Intent | None:
    """Fast path for slash commands — avoids a round-trip to GPT-5.1."""
    if not text:
        return None
    t = text.strip()
    if _HELP_RE.match(t):
        return Intent(action="help")
    if (m := _CREATE_RE.match(t)):
        return Intent(action="create_user", user_id=m.group(1))
    if (m := _USER_RE.match(t)):
        return Intent(action="select_user", user_id=m.group(1))
    if (m := _START_RE.match(t)):
        return Intent(action="start_run", topic=m.group(1).strip())
    if (m := _SET_RE.match(t)):
        return Intent(action="update_profile", profile_updates={m.group(1): m.group(2).strip()})
    if _STATUS_RE.match(t):
        return Intent(action="status")
    if _SKILLS_RE.match(t):
        return Intent(action="list_skills")
    if _MEMORIES_RE.match(t):
        return Intent(action="list_memories")
    if _TRAIN_RE.match(t):
        return Intent(action="train_planner")
    return None


# ============================================================ LLM intent


_INTENT_SYSTEM = """\
You are the intent router for **NanoResearch**, a conversational research
assistant. Classify the user's last message into one of the actions below
and extract any structured fields. Always return JSON only.

# Actions
- `help` — user wants to know what they can do
- `create_user` — start a new researcher profile.
   Fields: `user_id` (short identifier; default to a short, lowercase name like the user mentions)
- `select_user` — switch to an existing profile.
   Fields: `user_id`
- `update_profile` — change the active profile's fields.
   Fields: `profile_updates` — a flat map of `{field: value}`. Valid field keys:
     archetype, domain, research_preference, method_preference, risk_preference,
     baseline_strictness, resource_budget, feasibility_bias, writing_tone,
     claim_strength, section_organization, venue_style, persona_brief.
- `list_users` — list profiles
- `start_run` — begin a research run on the active profile.
   Fields: `topic` — the research topic in natural language (>=8 chars).
- `submit_feedback` — only valid when the run is awaiting feedback. The user's
   message itself is the feedback.
   Fields: `feedback` (verbatim user text)
- `status` — current run status
- `list_skills` / `list_memories` — inspect the active user's stores
- `train_planner` — run an SDPO update on buffered feedback
- `chitchat` — small talk, off-topic questions, ambiguous text. Provide a
   helpful conversational `reply` and gently steer back to research.

# Tips
- If the message is plainly a research topic and a profile is active, prefer
  `start_run` over `chitchat`.
- If the run is `awaiting_feedback`, **default** to `submit_feedback` unless
  the user clearly asks for something else (status, switch user, etc).
- `domain` values should be free-form, but bias toward broad fields like:
  Computer Science / Biology / Chemistry / Physics / Mathematics /
  Economics / Psychology / Medicine / Education / Linguistics / Sociology /
  Environmental Science / Engineering. Sub-fields are fine.
- Always include a short, friendly `reply` for the user — even when you also
  trigger a side-effect action.

# Output schema
{
  "action": "<one of the actions above>",
  "user_id": "...",          // when relevant
  "topic": "...",            // when relevant
  "feedback": "...",         // when relevant
  "profile_updates": {...},  // when relevant
  "reply": "..."             // always; what the assistant should say next
}
"""


def parse_with_llm(
    *,
    router: LLMRouter,
    text: str,
    session: IntentSession,
) -> Intent:
    """Ask GPT-5.1 to classify ``text`` given ``session`` context."""
    user_payload = {
        "session": session.model_dump(),
        "message": text,
    }
    res = router.complete(
        AgentRole.IDEATION,  # routed to Azure — same backend as everything else
        [
            ChatMessage(Role.SYSTEM, _INTENT_SYSTEM),
            ChatMessage(Role.USER, json.dumps(user_payload, ensure_ascii=False)),
        ],
        temperature=0.0,
        max_tokens=600,
        response_format={"type": "json_object"},
    )
    raw = _safe_json(res.text)
    if raw is None:
        _log.warning("intent_unparseable", text=text[:160])
        return Intent(
            action="chitchat",
            reply="Sorry — I couldn't make sense of that. Could you rephrase?",
        )
    try:
        intent = Intent.model_validate(raw)
    except Exception as e:  # noqa: BLE001
        _log.warning("intent_invalid", error=str(e), raw=str(raw)[:200])
        return Intent(
            action="chitchat",
            reply=str(raw.get("reply") or "I'm not sure how to handle that. Want to start a run?"),
        )
    return intent


def _safe_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    # Pull the outermost {...}
    a = text.find("{")
    b = text.rfind("}")
    if a >= 0 and b > a:
        try:
            v = json.loads(text[a : b + 1])
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ============================================================ public API


@dataclass
class ParsedIntent:
    intent: Intent
    source: Literal["local", "llm"]


def parse_intent(
    *,
    router: LLMRouter,
    text: str,
    session: IntentSession,
) -> ParsedIntent:
    local = parse_local(text)
    if local is not None:
        return ParsedIntent(local, "local")
    intent = parse_with_llm(router=router, text=text, session=session)
    return ParsedIntent(intent, "llm")

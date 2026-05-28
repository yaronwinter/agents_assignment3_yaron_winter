"""Per-user profile: tabular markdown that maps 1:1 to a RAM dict.

A profile has three sections:
  - categories: which dataset categories the user has asked about, with counts
  - intents:    which dataset intents the user has asked about, with counts
  - personal:   facts the user has explicitly stated about themselves
  - question:   the most recent question they asked (for context, not persisted)
  - route:      which routes were made, witth counts.

Storage: .profiles/{user_id}.md, written atomically via tmp + os.replace.

Updates are deterministic on the Python side. One LLM call per turn extracts
(category, intent, personal_details) from the exchange; Python increments the
counters and appends new personal facts. The LLM never sees the prior file
contents, so it cannot erase them.
"""
import os
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from agent.router import llm as base_llm
from agent.struct_tools import dataset

CATEGORIES = dataset.get_all_categories()
INTENTS = dataset.get_all_intents()

CATEGORY = "category"
INTENT = "intent"
PERSONAL = "personal"
QUESTION = "question"
ROUTE = "route"

PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    ".profiles",
)


def _path(user_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", user_id) or "default_user"
    return os.path.join(PROFILE_DIR, f"{safe}.md")

def _empty() -> Dict:
    return {CATEGORY: {}, INTENT: {}, ROUTE: {}, PERSONAL: [], QUESTION: []}


def _atomic_write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def load_profile(user_id: str) -> Dict:
    """Parse the markdown file into a dict. Returns the empty shape if no file."""
    path = _path(user_id)
    if not os.path.exists(path):
        return _empty()

    profile = _empty()
    section = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("## "):
                section = line[3:].strip()
                continue
            if not line.startswith("- "):
                continue
            body = line[2:].strip()
            if section in (CATEGORY, INTENT, ROUTE):
                if ":" not in body:
                    continue
                key, _, value = body.rpartition(":")
                key = key.strip()
                try:
                    profile[section][key] = int(value.strip())
                except ValueError:
                    continue
            elif section in (QUESTION, PERSONAL):
                profile[section].append(body)
    return profile


def save_profile(user_id: str, profile: Dict) -> None:
    """Serialize the dict back to markdown and write atomically."""
    lines = [f"# Profile", "", f"## {CATEGORY}"]
    for k, v in sorted(profile[CATEGORY].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {k}: {v}")
    lines += ["", f"## {INTENT}"]
    for k, v in sorted(profile[INTENT].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {k}: {v}")
    lines += ["", f"## {ROUTE}"]
    for k, v in sorted(profile[ROUTE].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {k}: {v}")
    lines += ["", f"## {PERSONAL}"]
    for fact in profile[PERSONAL]:
        lines.append(f"- {fact}")
    lines += ["", f"## {QUESTION}"]
    for q in profile[QUESTION]:
        lines.append(f"- {q}")

    _atomic_write(_path(user_id), "\n".join(lines) + "\n")


def render_profile(profile: Dict) -> str:
    """Human-readable rendering used both as LLM context and as the
    'what do you remember about me?' answer. Empty sections are omitted."""
    blocks = []
    if profile[PERSONAL]:
        blocks.append(
            "Personal facts:\n" + "\n".join(f"- {f}" for f in profile[PERSONAL])
        )
    if profile[QUESTION]:
        blocks.append(
            "Accumulated questions:\n" + "\n".join(f"- {q}" for q in profile[QUESTION])
        )
    if profile[CATEGORY]:
        top = sorted(profile[CATEGORY].items(), key=lambda x: x[1], reverse=True)
        blocks.append(
            "Categories the user has asked about:\n"
            + "\n".join(f"- {k}: {v}" for k, v in top)
        )
    if profile[INTENT]:
        top = sorted(profile[INTENT].items(), key=lambda x: x[1], reverse=True)
        blocks.append(
            "Intents the user has asked about:\n"
            + "\n".join(f"- {k}: {v}" for k, v in top)
        )
    if profile[ROUTE]:
        top = sorted(profile[ROUTE].items(), key=lambda x: x[1], reverse=True)
        blocks.append(
            "Routes the user has taken:\n"
            + "\n".join(f"- {k}: {v}" for k, v in top)
        )
    return "\n\n".join(blocks)


class Extraction(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="The dataset category the user's question is about, or None.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="The dataset intent the user's question is about, or None.",
    )
    personal_details: List[str] = Field(
        default_factory=list,
        description="New facts the user EXPLICITLY stated about themselves.",
    )


_extract_llm = base_llm.with_structured_output(Extraction)

_EXTRACT_PROMPT = f"""You analyze one exchange between a USER and a dataset assistant.

Extract three things and return them in the structured format:

1. category — which dataset category the user's question is about. Must be EXACTLY one of:
   {CATEGORIES}
   If the question is not about a specific category, return null.

2. intent — which dataset intent the user's question is about. Must be EXACTLY one of:
   {INTENTS}
   If the question is not about a specific intent, return null.

3. personal_details — a list of NEW personal facts the USER explicitly stated about THEMSELVES in their message. Include:
   - Name ("I'm Yaron")
   - Role/background ("I'm a data engineer")
   - Stated response-style preferences ("I prefer concise answers", "give me 3 examples at a time")
   - Stated topics of personal interest ("I'm particularly interested in X")
   Exclude:
   - Questions the user is asking (a question about refunds is NOT a personal fact)
   - Inferences or guesses about the user
   - Anything from the assistant's reply
   Return an empty list if no new personal facts were stated.

Do not infer. When uncertain, return null / empty list."""


def _extract(question: str, answer: str) -> Extraction:
    """Extract details that can characterize the user and their question, for profile updates and agent context."""
    e = _extract_llm.invoke([
        ("system", _EXTRACT_PROMPT),
        ("human", f"User: {question}\nAssistant: {answer}"),
    ])
    # Validate the LLM's category/intent against the known vocabulary so a
    # stray label (e.g. "REFUNDS" vs "REFUND") doesn't pollute the counters.
    if e.category and e.category not in CATEGORIES:
        e.category = None
    if e.intent and e.intent not in INTENTS:
        e.intent = None
    return e


def update_profile(user_id: str, question: str, answer: str, route: str) -> None:
    """Extract from this turn and persist. Linear, no special cases."""

    e = _extract(question, answer)
    p = load_profile(user_id)

    if e.category:
        p[CATEGORY][e.category] = p[CATEGORY].get(e.category, 0) + 1
    if e.intent:
        p[INTENT][e.intent] = p[INTENT].get(e.intent, 0) + 1
    for fact in e.personal_details:
        if fact not in p[PERSONAL]:
            p[PERSONAL].append(fact)
    if question not in p[QUESTION]:
        p[QUESTION].append(question)

    p[ROUTE][route] = p[ROUTE].get(route, 0) + 1

    save_profile(user_id, p)

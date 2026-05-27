"""Per-user profile storage.

A profile is a markdown file of distilled facts about one user (name, recurring
topics of interest, preferences). It is per-user — NOT per-session — so it
accumulates across all conversations the same user runs.

Storage: .profiles/{user_id}.md, written atomically via tmp + os.replace so a
Ctrl-C mid-write cannot corrupt the file.

Updates are performed by a small LLM call that merges the current profile with
the latest exchange. The merge prompt instructs the model to only add facts
the user has explicitly stated, never to infer.
"""
import os
import re

from agent.router import llm as merge_llm

PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    ".profiles",
)


def _path(user_id: str) -> str:
    # Defensive: strip anything that could escape PROFILE_DIR.
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", user_id) or "default_user"
    return os.path.join(PROFILE_DIR, f"{safe}.md")


def load_profile(user_id: str) -> str:
    path = _path(user_id)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _atomic_write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


MERGE_SYSTEM = """You maintain a concise profile of a user from their conversations with a dataset assistant.

Rules:
- Only record facts the user has EXPLICITLY stated about themselves
  (name, role, preferences for how to respond, recurring topics of interest).
- Do NOT infer or invent. If the latest turn reveals nothing new, return the
  current profile unchanged.
- Do NOT include conversation content or specific questions — only distilled facts.
- Keep the profile under 300 words.
- Use markdown with these sections (omit any section that would be empty):
  ## Identity
  ## Topics of interest
  ## Preferences
- Preserve existing facts unless directly contradicted; merge new ones in.

Return ONLY the updated profile in markdown. No preamble, no explanation."""


def _build_merge_input(current: str, question: str, answer: str) -> str:
    current_block = current if current.strip() else "(empty — no profile yet)"
    return (
        f"Current profile:\n{current_block}\n\n"
        f"Latest exchange:\nUser: {question}\nAssistant: {answer}\n\n"
        "Return the updated profile."
    )


def update_profile(user_id: str, question: str, answer: str) -> str:
    """Merge the latest turn into the user's profile and persist it.

    Returns the new profile content (may equal the old content if nothing
    new was extracted). Failures are swallowed — a profile update should
    never crash the main loop.
    """
    try:
        current = load_profile(user_id)
        merged = merge_llm.invoke([
            ("system", MERGE_SYSTEM),
            ("human", _build_merge_input(current, question, answer)),
        ]).content.strip()

        # Sanity: if the LLM returned something obviously broken (empty, or
        # much longer than our cap), prefer the existing profile.
        if not merged or len(merged) > 4000:
            return current

        if merged != current:
            _atomic_write(_path(user_id), merged)
        return merged
    except Exception:
        return load_profile(user_id)

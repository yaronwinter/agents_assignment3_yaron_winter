import argparse
import os
import uuid
os.environ["CLI_MODE"] = "1"

from rich.console import Console
from rich.panel import Panel
from agent.graph import app
from agent import profile as profile_store

console = Console()

def main():
    """
    The main function, which runs the agent in a loop,
    allowing the user to ask questions and receive answers until they choose to exit.

    Conversation history is persisted per --session (LangGraph thread_id) in
    .sessions/agent.sqlite, so re-running with the same session id restores
    the prior conversation.

    A separate per-user profile (distilled facts: name, recurring interests,
    preferences) is stored at .profiles/{user}.md and shared across all of
    a user's sessions. The profile is loaded into state before each turn
    and updated after each non-personal, non-OOS turn.
    """
    parser = argparse.ArgumentParser(description="BiText ReAct Agent")
    parser.add_argument("--session", type=str, default=uuid.uuid4().hex[:12])
    parser.add_argument("--user", type=str, default="default_user")
    args = parser.parse_args()

    config = {"configurable": {"thread_id": args.session}}

    console.print(
        Panel.fit(
            f"BiText ReAct Agent  (user: {args.user}, session: {args.session})",
            style="bold blue"
        )
    )

    while True:
        # Get user input
        question = input("\nUser > ")
        question = question.strip()
        if len(question) == 0:
            continue

        # Exit condition
        if question.lower() in ["exit", "quit"]:
            break

        # Load + render the profile fresh each turn so any update from the
        # previous turn (or a manual edit on disk) is picked up immediately.
        profile_text = profile_store.render_profile(
            profile_store.load_profile(args.user)
        )

        # Invoke the agent. The checkpointer rehydrates prior state for this
        # thread_id; iterations is reset per turn so MAX_ITERATIONS guards a
        # single invocation, not the lifetime of the session.
        result = app.invoke(
            {
                "question": question,
                "iterations": 0,
                "profile": profile_text,
            },
            config=config,
        )

        # Display the agent's answer
        console.print(
            Panel(
                result["answer"],
                title="Agent"
            )
        )

        # Always update — extraction returns empty for OOS / pure-meta turns,
        # making the dict update a no-op. No route-based special-casing.
        profile_store.update_profile(args.user, question, result["answer"], result.get("route"))


if __name__ == "__main__":
    main()

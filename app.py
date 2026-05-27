import argparse
import os
os.environ["CLI_MODE"] = "1"

from rich.console import Console
from rich.panel import Panel
from agent.graph import app

console = Console()

def main():
    """
    The main function, which runs the agent in a loop,
    allowing the user to ask questions and receive answers until they choose to exit.

    Conversation history is persisted per --session (LangGraph thread_id) in
    .sessions/agent.sqlite, so re-running with the same session id restores
    the prior conversation.
    """
    parser = argparse.ArgumentParser(description="BiText ReAct Agent")
    parser.add_argument(
        "--session",
        default="default",
        help="Session id; same value across runs restores the same conversation.",
    )
    args = parser.parse_args()

    config = {"configurable": {"thread_id": args.session}}

    console.print(
        Panel.fit(
            f"BiText ReAct Agent  (session: {args.session})",
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

        # Invoke the agent. The checkpointer rehydrates prior state for this
        # thread_id; iterations is reset per turn so MAX_ITERATIONS guards a
        # single invocation, not the lifetime of the session.
        result = app.invoke(
            {
                "question": question,
                "iterations": 0,
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


if __name__ == "__main__":
    main()

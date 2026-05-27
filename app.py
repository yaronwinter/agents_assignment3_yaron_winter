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
    """
    console.print(
        Panel.fit(
            "BiText ReAct Agent",
            style="bold blue"
        )
    )

    while True:
        # Get user input
        question = input("\nUser > ")

        # Exit condition
        if question.lower() in ["exit", "quit"]:
            break

        # Invoke the agent with the user's question and an initial state
        result = app.invoke({
            "question": question,
            "iterations": 0,
            "messages": [],
            "answer": "",
            "route": ""
        })

        # Display the agent's answer
        console.print(
            Panel(
                result["answer"],
                title="Agent"
            )
        )


if __name__ == "__main__":
    main()

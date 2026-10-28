"""
AutoGen HR Assistant Agent
Asks an AssistantAgent (system message: "You are an HR assistant.")
for a 5-line job description for a Junior Python Developer.
"""

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

# Load OPENAI_API_KEY from the .env file into the environment
load_dotenv()


TASK = (
    "Write a job description for a Junior Python Developer. "
    "Format it as a one-line title 'Junior Python Developer', followed by "
    "exactly 5 numbered lines (1. to 5.), each a single sentence. "
    "Do not add any introduction, closing remarks, or extra lines."
)


async def main() -> None:
    # Read the API key from the environment - never hard-code it
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not found. Add it to your .env file and try again.")

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

    agent = AssistantAgent(
        name="hr_assistant",
        model_client=model_client,
        system_message="You are an employment lawyer.",
    )

    try:
        await Console(agent.run_stream(task=TASK))
    finally:
        # Always close the client, otherwise the script can hang
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
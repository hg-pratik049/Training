import os
import asyncio
from pathlib import Path
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo  # only needed for some Gemini variants

load_dotenv()
def _get_gemini_key() -> str:
    """
    Return Gemini API key from common env var names.
    Prefers GOOGLE_API_KEY, then GOOGLE_GEMINI_API_KEY, then GEMINI_API_KEY.
    """
    key = (
        
        os.getenv("GEMINI_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "Gemini API key not found. Set one of: GOOGLE_API_KEY, "
            "GOOGLE_GEMINI_API_KEY, or GEMINI_API_KEY."
        )
    return key


async def main() -> None:
    # --- Model client: Gemini via OpenAI-compatible client
   


    model_client = OpenAIChatCompletionClient(
    # Pick a model that your account/project actually has access to
    model="gemini-2.5-flash-lite",  # try this first; 'flash-lite' may not be listed on your account
    api_key=_get_gemini_key(),  # your existing helper

    # 1) CRITICAL: point to Google's OpenAI-compatible endpoint
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # <-- required

    # 2) Because this isn’t an OpenAI-native model, tell AutoGen the model’s capabilities
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        structured_output=True,
        family="unknown",
    ),

    # Optional: give yourself a little resiliency on transient errors
    max_retries=3,
)

    # --- Agents
    coder = AssistantAgent(
        "coder",
        model_client=model_client,
        system_message=(
            "You are a senior engineer. Think step-by-step, then output ONLY runnable "
            "Python inside ```python``` blocks—no commentary."
        ),
    )

    # Local (non-Docker) executor
    runs_dir = Path.cwd() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    local_executor = LocalCommandLineCodeExecutor(work_dir=runs_dir)

    executor = CodeExecutorAgent(
        "executor",
        model_client=model_client,     # allows reflection/narration of results
        code_executor=local_executor,  # REQUIRED: pass a CodeExecutor
    )

    user = UserProxyAgent("user")  # human-in-the-loop

    termination = TextMentionTermination("exit", sources=["user"])
    team = RoundRobinGroupChat([user, coder, executor], termination_condition=termination)

    try:
        # Interactive console; type 'exit' to stop
        await Console(team.run_stream())
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
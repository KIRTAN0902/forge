"""Interactive Human Question Tool (prompts user with options like Claude Code / Codex)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from forge.policies.types import Permission, RiskLevel
from forge.tools.base import Tool, ToolInput, ToolOutput
from forge.tools.environment import ToolEnvironment


class AskUserQuestionInput(ToolInput):
    question: str = Field(description="The question to ask the user.")
    options: list[str] = Field(
        default_factory=list,
        description="Selectable numbered choices presented to the user.",
    )
    default_option: str | None = Field(
        default=None,
        description="Default choice if the user presses Enter without typing.",
    )


class AskUserQuestionOutput(ToolOutput):
    answer: str = Field(description="The user's selected choice or typed response.")


class AskUserQuestion(Tool[AskUserQuestionInput, AskUserQuestionOutput]):
    """Prompts the user with a question and numbered options, pausing until answered."""

    name: ClassVar[str] = "ask_user_question"
    description: ClassVar[str] = (
        "Ask the user an interactive clarifying question with selectable numbered options. "
        "Blocks and waits for the user's choice or custom text response."
    )
    input_model = AskUserQuestionInput
    output_model = AskUserQuestionOutput
    risk = RiskLevel.LOW
    permissions = frozenset({Permission.READ})

    async def run(self, args: AskUserQuestionInput, env: ToolEnvironment) -> AskUserQuestionOutput:
        handler = getattr(env, "ask_question", None)
        if callable(handler):
            ans = await handler(args.question, args.options, args.default_option)
            return AskUserQuestionOutput(answer=str(ans))

        fallback = args.default_option or (args.options[0] if args.options else "Proceed")
        return AskUserQuestionOutput(answer=fallback)

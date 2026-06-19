from pydantic import BaseModel, Field

class CodeAction(BaseModel):
    """Structured action emitted by the root RLM model: code to run in the REPL."""

    code: str = Field(description="Executable Python code, without markdown fences")


class SubAnswer(BaseModel):
    """Structured answer returned by a sub-model call."""

    answer: str = Field(description="The sub-model's answer in plain text")

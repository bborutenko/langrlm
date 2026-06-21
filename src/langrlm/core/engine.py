import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langrlm.core.prompts import SYSTEM_PROMPT, FORCE_ANSWER_PROMPT
from langrlm.core.repl import REPL
from langrlm.core.context_store import BaseContextStore
from langrlm.utils.schemas import CodeAction


class Engine:
    """Core RLM loop that drives the models against the REPL.

    Runs the recursive interaction: the root model writes Python code, the code
    is executed in a sandboxed :class:`REPL` (where it can slice the context and
    call the sub-model via ``llm_query``), and the resulting output is fed back
    until the model submits an answer or the depth budget is exhausted. Usually
    constructed by :class:`~langrlm.core.rlm.RLM` rather than directly.
    """

    rlm_model: BaseChatModel
    sub_model: BaseChatModel
    context: BaseContextStore
    max_depth: int
    rlm_structured: bool
    sub_structured: bool

    def __init__(
            self,
            rlm_model: BaseChatModel,
            sub_model: BaseChatModel,
            context: BaseContextStore,
            max_depth: int,
            rlm_structured: bool = False,
            sub_structured: bool = False,
        ):
        """Build the engine.

        Args:
            rlm_model: Root model that orchestrates the task by writing REPL code.
            sub_model: Model used for recursive sub-calls (``llm_query``).
            context: Data the agent reasons over, exposed to the REPL.
            max_depth: Maximum number of REPL steps before the model is forced
                to answer.
            rlm_structured: Whether ``rlm_model`` supports structured output. If
                True, its code is requested via a schema instead of being parsed
                out of free text.
            sub_structured: Whether ``sub_model`` supports structured output.
                Stored for use by sub-model calls.
        """
        self.rlm_model = rlm_model
        self.sub_model = sub_model
        self.context = context
        self.max_depth = max_depth

        self.rlm_structured = rlm_structured
        self.sub_structured = sub_structured

        self._structured_rlm = (
            rlm_model.with_structured_output(CodeAction) if rlm_structured else None
        )

    def _llm_query(self, prompt: str, text: str) -> str:
        response = self.sub_model.invoke(f"{prompt}\n\n{text}")
        return self._to_text(response.content)

    def complete(self, message: str) -> str:
        """Run the full RLM loop for a task and return the final answer.

        Loads the context, spins up a fresh REPL, then repeatedly asks the root
        model for a step of code and executes it. Each step the model can read
        the context and query the sub-model; the REPL output is appended back to
        the conversation. The loop ends when the model calls ``SUBMIT(...)``
        (or, in text mode, replies without a code block). If ``max_depth`` steps
        pass without an answer, the model is prompted once more to answer with
        what it has.

        Args:
            message: The task/question to solve over the context.

        Returns:
            The final answer as a string.
        """
        self.context.load()

        repl = REPL(context_store=self.context, llm_query_fn=self._llm_query)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]

        for _ in range(self.max_depth):
            if self.rlm_structured:
                action = self._structured_rlm.invoke(messages)
                code = action.code
                response = AIMessage(content=f"```python\n{code}\n```")
            else:
                response = self.rlm_model.invoke(messages)
                text = self._to_text(response.content)
                code = self._extract_code(text)

                if code is None:
                    return text

            output, done = repl.execute(code)
            if done:
                return output

            messages.append(response)
            messages.append(HumanMessage(content=f"REPL output:\n{output}"))

        messages.append(HumanMessage(content=FORCE_ANSWER_PROMPT))
        final = self.rlm_model.invoke(messages)
        return self._to_text(final.content)

    def _extract_code(self, text: str) -> str | None:
        match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    def _to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return str(content)

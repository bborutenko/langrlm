import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from langrlm.core.prompts import SYSTEM_PROMPT, FORCE_ANSWER_PROMPT
from langrlm.core.repl import REPL
from langrlm.core.context_store import BaseContextStore


class Engine:
    rlm_model: BaseChatModel
    sub_model: BaseChatModel
    context: BaseContextStore
    max_depth: int

    def __init__(
            self,
            rlm_model: BaseChatModel,
            sub_model: BaseChatModel,
            context: BaseContextStore,
            max_depth: int,
        ):
        self.rlm_model = rlm_model
        self.sub_model = sub_model
        self.context = context
        self.max_depth = max_depth

    def _llm_query(self, prompt: str, text: str) -> str:
        response = self.sub_model.invoke(f"{prompt}\n\n{text}")
        return self._to_text(response.content)

    def complete(self, message: str) -> str:
        self.context.load()

        repl = REPL(context_store=self.context, llm_query_fn=self._llm_query)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]

        for _ in range(self.max_depth):
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

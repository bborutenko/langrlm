from langchain_core.language_models.chat_models import BaseChatModel

from langrlm.core.context_store import BaseContextStore
from langrlm.core.engine import Engine
from langrlm.utils.schemas import CodeAction, SubAnswer


class RLM:
    engine: Engine
    _has_structured_output: bool

    def __init__(
            self,
            model: BaseChatModel,
            context: BaseContextStore,
            sub_model: BaseChatModel | None = None,
            max_depth: int = 10,
        ):
        sub_model = sub_model or model

        rlm_structured = self._supports_structured_output(model)
        sub_structured = self._supports_structured_output(sub_model)

        self._has_structured_output = rlm_structured

        self.code_schema = CodeAction if rlm_structured else None
        self.sub_schema = SubAnswer if sub_structured else None

        self.engine = Engine(
            rlm_model=model,
            sub_model=sub_model,
            context=context,
            max_depth=max_depth,
            rlm_structured=rlm_structured,
            sub_structured=sub_structured,
        )

    @staticmethod
    def _supports_structured_output(model: BaseChatModel) -> bool:
        # Building the structured wrapper raises NotImplementedError for models
        # that don't support it — no network call needed.
        try:
            model.with_structured_output(CodeAction)
            return True
        except NotImplementedError:
            return False

    def invoke(self, message: str) -> str:
        return self.engine.complete(message)

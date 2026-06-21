from langchain_core.language_models.chat_models import BaseChatModel

from langrlm.core.context_store import BaseContextStore
from langrlm.core.engine import Engine
from langrlm.utils.schemas import CodeAction, SubAnswer


class RLM:
    """Recursive Language Model agent built on top of a LangChain chat model.

    Wraps a LangChain model and drives it as an RLM: the model writes Python
    code that is executed in a sandboxed REPL, where it can read the provided
    context in slices and call a sub-model on those slices, recursing until it
    submits a final answer.
    """

    engine: Engine
    _has_structured_output: bool

    def __init__(
            self,
            model: BaseChatModel,
            context: BaseContextStore,
            sub_model: BaseChatModel | None = None,
            max_depth: int = 10,
        ):
        """Build an RLM agent.

        Args:
            model: The root LangChain chat model. It orchestrates the task by
                writing the REPL code on each step.
            context: The data the agent reasons over (e.g. a large document).
                Exposed to the model's code via ``context_slice`` /
                ``context_size``. See :func:`create_context_store`.
            sub_model: Optional cheaper/faster model used for the recursive
                sub-calls (``llm_query``). Defaults to ``model`` when omitted.
            max_depth: Maximum number of REPL steps the root model may take
                before it is forced to produce a final answer. Acts as the
                recursion/iteration budget.
        """
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
        """Run the agent on a task and return its final answer.

        Args:
            message: The task/question for the agent to solve over the context.

        Returns:
            The final answer as a string, produced either when the model calls
            ``SUBMIT(...)`` in its REPL code or when ``max_depth`` is reached
            and the model is forced to answer.
        """
        return self.engine.complete(message)

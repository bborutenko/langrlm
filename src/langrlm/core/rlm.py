from langchain_core.language_models.chat_models import BaseChatModel

from langrlm.core.context_store import BaseContextStore
from langrlm.core.engine import Engine


class RLM:
    engine: Engine
    
    def __init__(
            self,
            model: BaseChatModel,
            context: BaseContextStore,
            sub_model: BaseChatModel | None = None,
            max_depth: int = 10,
        ):
        self.engine = Engine(
            rlm_model=model,
            sub_model=sub_model or model,
            max_depth=max_depth,
            context=context
        )
        
    def invoke(self, message: str) -> str:
        return self.engine.complete(message)
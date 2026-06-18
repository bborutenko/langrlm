from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from langrlm.core.context_store import BaseContextStore
from langrlm.core.engine import Engine


class RLM:
    engine: Engine
    
    def __init__(
            self, 
            model: BaseChatModel, 
            sub_model: BaseChatModel,
            context: BaseContextStore,
            max_depth: int
        ):
        self.engine = Engine(
            rlm_model=model, 
            sub_model=sub_model, 
            max_depth=max_depth,
            context=context
        )
        
    def invoke(self, message: BaseMessage):
        pass
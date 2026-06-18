from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

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

    def query(self, message: BaseMessage):
        pass
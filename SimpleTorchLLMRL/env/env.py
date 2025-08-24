

from abc import ABC, abstractmethod
from ast import Tuple
from typing import Any, List

from SimpleTorchLLMRL.chat.message import Message, Rollout, MessageType
from SimpleTorchLLMRL.utils.typing import State



class Env(ABC):



    def __init__(self) -> None:
        pass


    @abstractmethod
    def setup(self,  *args: Any, **kwargs: Any) -> tuple[List[Message], State]: pass

        
    @abstractmethod
    def get_env_response(self, rollout: Rollout, state: State) -> tuple[Message, State]: pass


    @abstractmethod
    def is_complete(self, rollout:Rollout, state: State)-> bool: pass


    def _reached_max_turns(self,rollout: Rollout, max_turns: int) -> bool:

        user_messages = [message for message in rollout if message.type == MessageType.MESSAGE]
        if len(user_messages) > max_turns:
            return False
        return True
        












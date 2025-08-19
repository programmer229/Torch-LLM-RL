

from abc import ABC, abstractmethod
from ast import Tuple
from typing import List

from AgentOrchestration.chat.message import Message, Rollout
from AgentOrchestration.utils.typing import State



class Env(ABC):



    def __init__(self) -> None:
        pass


    @abstractmethod
    def setup(self) -> Tuple(List[Message], State): pass

        
    @abstractmethod
    def get_env_response(self, rollout: Rollout, state: State) -> Message: pass


    @abstractmethod
    def is_complete(self, state: State): pass


    def _reached_max_turns(self,rollout: Rollout, max_turns: int) -> bool:

        user_messages = [message for message in rollout if message.type == MessageType.Message]
        if len(user_messages) > max_turns:
            return False
        return True
        














from abc import ABC, abstractmethod

from AgentOrchestration.chat.message import Message




class Env(ABC):



    def __init__(self, 
        max_turns:int = None) -> None:
        self.max_truns = None

    @abstractmethod
    def get_sys_prompt(self) -> Message: pass
    

    @abstractmethod
    def get_inital_prompt(self) -> Message: pass
        
    @abstractmethod
    def get_response_to_model(self) -> Message: pass


    def _over_max_turns(self,rollout) ->:

        user_messages = [message for message in rollout if message.type = MessageType.Message]
        if len(user_messages) > _over_max_turns:
            return False
        return True
        












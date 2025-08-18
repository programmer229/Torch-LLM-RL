

from abc import ABC, abstractmethod

from AgentOrchestration.chat.message import Message




class Env(ABC):



    def __init__(self) -> None:
        self.env_complete = False

    @abstractmethod
    def get_sys_prompt(self) -> Message: pass
    

    @abstractmethod
    def get_inital_prompt(self) -> Message: pass
        
    @abstractmethod
    def get_response_to_model(self) -> Message: pass


        












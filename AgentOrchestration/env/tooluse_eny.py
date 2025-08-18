

from .env import Env




class ToolUseEnv(Env):



    def __init__(self, tools) -> None:
        super().__init__()

    

    @abstractmethod
    def get_inital_prompt(self) -> Message: pass
        
    @abstractmethod
    def get_response_to_model(self) -> Message: pass

    
    
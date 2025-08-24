

from typing import Optional, List

from .env import Env
from .QASolve import QASolverEnv
from SimpleTorchLLMRL.chat.message import Message, MessageType, Rollout
from SimpleTorchLLMRL.reward.reward import Reward
from SimpleTorchLLMRL.tools.tool import  Tool
from SimpleTorchLLMRL.tools.toolmanager import ToolManger
from SimpleTorchLLMRL.utils.typing import State

class ToolUseEnv(QASolverEnv):



    def __init__(self,
                tools:list[Tool]| None = None,
                custom_sys_prompt = None) -> None:
        super().__init__()
        
        self.tool_manager = ToolManger(tools=(tools or []))
        self.custom_sys_prompt = (custom_sys_prompt or "") + self.tool_manager.tool_explanation_prompt()


    def get_response_to_model(self, rollout: Rollout, state: State) -> tuple[list[Message], State]: 
        
        most_recent_mesaage = rollout[-1]
        
        message = self.tool_manager.process(most_recent_mesaage)
        return [message], state



            







    
    


from typing import Optional
from .env import Env

from AgentOrchestration.chat.message import Message, MessageType
from AgentOrchestration.reward.reward import Reward
from AgentOrchestration.tools.tool import ToolManger


class ToolUseEnv(Env):



    def __init__(self, tools, dataset, 
                reward: Reward,
                tools:ToolManger= None,
                custom_sys_prompt = None) -> None:
        super().__init__()
        self.reward = reward
        self.tool_manager = ToolManger
        self.dataset = dataset
        self.custom_sys_prompt = ""
    
    def get_sys_prompt(self) -> Message:
        
        tool_explanation = self.ToolManger.get_model_tool_explanation() if self.ToolManger else ""

        sys_prompt = self.custom_sys_prompt + tool_explanation     
        message = Message(content=sys_prompt, type=Message.SYSTEM)
        return message

    def get_inital_prompt(self) -> Message: 
        
        # self.dataset 
        #TODO implement the dataset getting

        return message


    def get_response_to_model(self, rollout: Rollout) -> Optional[Message]: 
        
        most_recent_mesaage = rollout[-1]

        if self.tool_manager:
            #Check if Tool Call
            ...

        else:
            #What we return here a little tricky
            
            return None






    
    
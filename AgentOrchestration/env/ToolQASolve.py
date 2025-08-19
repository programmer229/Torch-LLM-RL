

from typing import Optional, List
from .env import Env

from AgentOrchestration.chat.message import Message, MessageType
from AgentOrchestration.reward.reward import Reward
from AgentOrchestration.tools.tool import ToolManger, Tool


class ToolUseEnv(Env):



    def __init__(self, tools, dataset, 
                reward: Reward,
                tools:List[Tool]= None,
                custom_sys_prompt = None) -> None:
        super().__init__()
        self.reward = reward
        self.tool_manager = ToolManger(tools=tools)
        self.dataset = dataset
        self.custom_sys_prompt = ""
    
    def get_sys_prompt(self) -> Message:
        
        

        sys_prompt = self.custom_sys_prompt + tool_explanation     
        message = Message(content=sys_prompt, type=Message.SYSTEM)
        return message

    def get_inital_prompt(self) -> Message: 
        
        # self.dataset 
        #TODO implement the dataset getting

        return message


    def get_response_to_model(self, rollout: Rollout) -> Optional[Message]: 
        
        most_recent_mesaage = rollout[-1]






    
    
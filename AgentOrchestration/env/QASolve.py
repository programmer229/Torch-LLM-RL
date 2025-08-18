

from typing import Optional

from AgentOrchestration.reward import reward
from .env import Env

from AgentOrchestration.chat.message import Message, MessageType
from AgentOrchestration.reward.reward import Reward
from AgentOrchestration.tools.tool import ToolManger


class ToolUseEnv(Env):



    def __init__(self, tools, dataset, 
                reward: Reward,
                tools:ToolManger= None,
                custom_sys_prompt = None,
                max_turns:int = None) -> None:
        super().__init__()
        self.reward = reward
        
        self.dataset = dataset
        self.custom_sys_prompt = ""
        
        self.question = None
        self.ground_truth = None
        
    
    def get_sys_prompt(self) -> Message:
        
        

        sys_prompt = self.custom_sys_prompt + tool_explanation     
        message = Message(content=sys_prompt, type=Message.SYSTEM)
        return message

    def get_inital_prompt(self) -> Message: 
        
        # self.dataset 
        #TODO implement the dataset getting

        return message


    def get_response_to_model(self, rollout: Rollout) -> None: 
        #TODO This one is complex hey cause we can either return a message or the reward
            #Maybe we have a property on Rollout is complete and reward int?
                #adds the new rollout in place is that bad?
        most_recent_mesaage = rollout[-1]
        rollout.is_complete = True
        rollout.reward = reward(model_solution= most_recent_message, ground_truth = self.ground_truth) 
        #ahh this get's complex cause now we have to think about the question assigned. 
        # Now probably makes more sense to have unique env instance for each
        #Right so you have to use some kind of hybrid system so one instance of env per rollout
        #this makes sense since env should be stateful

        






    
    
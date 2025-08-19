

from typing import Optional, List, Tuple


from AgentOrchestration.reward import reward
from .env import Env

from AgentOrchestration.chat.message import Message, MessageType, Rollout
from AgentOrchestration.reward.reward import Reward
# from AgentOrchestration.tools.tool import ToolManger
from AgentOrchestration.utils.typing import State


class QASolverEnv(Env):



    def __init__(self, 
                custom_sys_prompt = None
                ) -> None:
        super().__init__()
     
        self.custom_sys_prompt = ""

        
    
    def setup(self, question, ground_truth) -> Tuple[List[Message], State]: 
        
        sys_prompt = self.custom_sys_prompt
        sys_message = Message(content=sys_prompt, type=MessageType.SYSTEM)
        
        state = {}
        state["question"] = question
        state["ground_truth"] = ground_truth
        
        prompt = Message(content=f"Solve the following. Output Answer in Boxed. {question}",type=MessageType.MESSAGE)

        messages = [sys_message, prompt]

        return (messages, state)
        
    # self.dataset 
        #TODO implement the dataset getting

        # return message

    def is_complete(self, rollout:Rollout, state: State) -> bool:

        return self._reached_max_turns(rollout, max_turns=1) # single turn


    def get_env_response(self, rollout:Rollout, state: State) -> Message: 
        #No response needed for this class
        pass


        
        



        #ahh this get's complex cause now we have to think about the question assigned. 
        # Now probably makes more sense to have unique env instance for each
        #Right so you have to use some kind of hybrid system so one instance of env per rollout
        #this makes sense since env should be stateful

        






    
    
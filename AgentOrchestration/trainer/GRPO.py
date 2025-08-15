

from abc import ABC, abstractmethod
import token
from typing import List


from .train import Trainer
from AgentOrchestration.utils.message import Rollouts, MessageType
import torch

class GRPO(Trainer):



    def __init__(self, model, tokenizer, eps) -> None:
        super().__init__()
        self.model = model
        self.ref_model = model.copy()
        self.tokenizer = tokenizer
        self.eps = eps

        

    def _calculate_advantage(self, rewards: torch.tensor) -> torch.tensor:
        advantage = torch.tensor([])

        float_rewards = rewards.float()
        reward_mean = torch.mean(float_rewards)
        reward_std = torch.std(float_rewards)

        advantage = (float_rewards -reward_mean)/reward_std
        advantage= torch.nan_to_num(advantage, 0)
        
        return advantage
    

    
    @abstractmethod
    def calculate_loss(self, rollouts: List[Rollouts], rewards: torch.TensorType):
        
        
        backprop_message = [message for rollout in rollouts for message in rollout if message.type ==  MessageType.Model]
        
        
        
        advantage = self._calculate_advantage()
        

        model_probs = torch.tensor([])
        ref_model_probs = torch.tensor([])
        


        formatted_string = torch.tensor([rollout.format_conversation_str for rollout in rollouts])
        inputs_ids = torch.tensor([self.tokenizer.encode(string) for string in  formatted_string])
        
        
        
        model_logits = torch.concat(self.model(inputs_ids))
        ref_model_logits = torch.concat(self.ref_model(inputs_ids))
        
        model_probs = torch.log(model_logits)
        ref_model_probs = torch.log(ref_model_logits)

        ratios = model_probs/ ref_model_probs

        clipped_min = torch.clip(ratios, 1-self.eps, 1+self.eps)

        loss = torch.min(ratios, clipped_min)

        return loss
        






            



    
    
    




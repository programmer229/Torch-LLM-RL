
import copy
from abc import ABC, abstractmethod
import token
from typing import List
from collections import defaultdict

from .trainer import Trainer
from AgentOrchestration.chat.message import Rollout, MessageType
import torch

class GRPO(Trainer):



    def __init__(self, model, tokenizer, eps) -> None:
        super().__init__()
        self.model = model
        self.ref_model = copy.deepcopy(model) if model else None
        self.tokenizer = tokenizer
        self.eps = eps

        

    def _calculate_advantage(self, rewards: torch.tensor) -> torch.tensor:
        

        if len(rewards) <= 1:
            return torch.clip(rewards, min=-1, max=1)

        advantage = torch.tensor([])

        float_rewards = rewards.float()
        reward_mean = torch.mean(float_rewards)
        reward_std = torch.std(float_rewards)

        advantage = (float_rewards -reward_mean)/reward_std
        advantage= torch.nan_to_num(advantage, 0)
    
        return advantage
    

    def _calculate_prob_ratios(self, inputs_ids):

        model_logits = self.model(inputs_ids).logits
        ref_model_logits = self.ref_model(inputs_ids).logits
        softmax = torch.nn.Softmax(dim=-1)
        model_probs = softmax(model_logits)
        ref_model_probs = softmax(ref_model_logits)

        ratios = model_probs/ ref_model_probs

      
        return ratios

    
    def calculate_loss(self, rollouts: List[Rollout], rewards: torch.TensorType):
        
        
        # Get indices of MODEL message tokens for backprop only
        backprop_indices = defaultdict(list)
        for rollout_idx, rollout in enumerate(rollouts):
            token_pos = 0
            for msg in rollout:
                msg_tokens = self.tokenizer.encode(msg.content)
                if msg.type == MessageType.MODEL:
                    # Store (token_start, token_end) for each MODEL message in this rollout
                    backprop_indices[rollout_idx].append((token_pos, token_pos + len(msg_tokens)))
                token_pos += len(msg_tokens)
        
        
        
        advantage = self._calculate_advantage(rewards)

        formatted_strings = [rollout.format_conversation_str() for rollout in rollouts]
        encoded_inputs = [self.tokenizer.encode(string) for string in formatted_strings]
        
        # Handle empty rollouts case
        if not encoded_inputs:
            return torch.tensor(0.0)
        
        # Pad sequences to same length
        max_length = max(len(seq) for seq in encoded_inputs)
        inputs_ids = torch.tensor([
            seq + [self.tokenizer.pad_token_id] * (max_length - len(seq)) 
            for seq in encoded_inputs
        ])
        
        
        ratios = self._calculate_prob_ratios(inputs_ids)
        
        # Calculate loss per rollout, handling multiple MODEL messages per rollout
        rollout_losses = []
        
        for rollout_idx in range(len(rollouts)):
            rollout_advantage = advantage[rollout_idx]
            model_message_losses = []
            
            # Get all MODEL message token ranges for this rollout
            if rollout_idx in backprop_indices:
                for start_pos, end_pos in backprop_indices[rollout_idx]:
                    # Get ratios for this MODEL message's tokens
                    message_ratios = ratios[rollout_idx, start_pos:end_pos]
                    
                    if len(message_ratios) > 0:
                        clipped_ratios = torch.clip(message_ratios, 1-self.eps, 1+self.eps)
                        
                        # GRPO/PPO loss for this MODEL message
                        loss_unclipped = rollout_advantage * message_ratios
                        loss_clipped = rollout_advantage * clipped_ratios
                        loss_per_token = torch.min(loss_unclipped, loss_clipped)
                        
                        # Average loss over tokens in this MODEL message
                        message_loss = loss_per_token.mean()
                        model_message_losses.append(message_loss)
            
            # Average loss over all MODEL messages in this rollout
            if model_message_losses:
                rollout_loss = torch.stack(model_message_losses).mean()
                rollout_losses.append(rollout_loss)
        
        if rollout_losses:
            # Average loss across all rollouts
            loss = -torch.stack(rollout_losses).mean()  # Negative for gradient descent
        else:
            loss = torch.tensor(0.0)

        return loss
        






            



    
    
    





import copy
from abc import ABC, abstractmethod
import token
from typing import List


from .trainer import Trainer
from AgentOrchestration.utils.message import Rollout, MessageType
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
            raise ValueError("Rollouts much be more than 1 inorder to calculate the adnvatege")

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
        backprop_indices = []
        for rollout_idx, rollout in enumerate(rollouts):
            token_pos = 0
            for msg in rollout.messages:
                msg_tokens = self.tokenizer.encode(msg.content)
                if msg.type == MessageType.MODEL:
                    # Store (batch_idx, token_start, token_end) for MODEL messages
                    backprop_indices.append((rollout_idx, token_pos, token_pos + len(msg_tokens)))
                token_pos += len(msg_tokens)
        
        
        
        advantage = self._calculate_advantage(rewards)

        formatted_strings = [rollout.format_conversation_str() for rollout in rollouts]
        encoded_inputs = [self.tokenizer.encode(string) for string in formatted_strings]
        
        # Pad sequences to same length
        max_length = max(len(seq) for seq in encoded_inputs)
        inputs_ids = torch.tensor([
            seq + [self.tokenizer.pad_token_id] * (max_length - len(seq)) 
            for seq in encoded_inputs
        ])
        
        
        ratios = self._calculate_prob_ratios(inputs_ids)
        
        # Calculate loss per rollout, then average across rollouts
        rollout_losses = []
        
        for i, (batch_idx, start_pos, end_pos) in enumerate(backprop_indices):
            # Get ratios for this rollout's MODEL tokens
            rollout_ratios = ratios[batch_idx, start_pos:end_pos]
            rollout_advantage = advantage[i]  # Advantage for this rollout
            
            if len(rollout_ratios) > 0:
                clipped_ratios = torch.clip(rollout_ratios, 1-self.eps, 1+self.eps)
                
                # GRPO/PPO loss for this rollout
                loss_unclipped = rollout_advantage * rollout_ratios
                loss_clipped = rollout_advantage * clipped_ratios
                loss_per_token = torch.min(loss_unclipped, loss_clipped)
                
                # Average loss over tokens in this rollout
                rollout_loss = loss_per_token.mean()
                rollout_losses.append(rollout_loss)
        
        if rollout_losses:
            # Average loss across all rollouts
            loss = -torch.stack(rollout_losses).mean()  # Negative for gradient descent
        else:
            loss = torch.tensor(0.0)

        return loss
        






            



    
    
    






import torch
from typing import Protocol, List

from SimpleTorchLLMRL.chat.message import Rollout, MessageType

def length_penalty(rollouts: List[Rollout], ground_truth: List[str]) -> torch.TensorType:
    rewards = []
    
    for rollout in rollouts:
        # Get the last model message from the rollout
        model_message = None
        for message in reversed(rollout._messages):
            if message.type == MessageType.MODEL:
                model_message = message
                break
        
        if model_message is not None:
            # Calculate penalty based on length difference from target (20 chars)
            length_diff = abs(len(model_message.content) - 20)
            reward = 1 -length_diff / 100.0  # Negative penalty, normalized
            rewards.append(reward)
        else:
            # No model message found, give zero reward
            rewards.append(0.0)
    
    return torch.tensor(rewards)
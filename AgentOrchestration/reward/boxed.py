
import torch
from torch import TensorType
from typing import List
import re

from AgentOrchestration.chat.message import Rollout, MessageType



class BoxedReward:

    def __init__(self, format_reward = 0) -> None:
        self.format_reward = format_reward

    def __call__(self, rollouts: List[Rollout], ground_truth: List[str]) -> TensorType:
        rewards = []
        for i, rollout in enumerate(rollouts):
            # Extract content from boxed notation in MODEL messages
            boxed_content = None
            for message in rollout:
                if message.type == MessageType.MODEL:
                    # Look for \boxed{...} pattern
                    match = re.search(r'\\boxed\{([^}]*)\}', message.content)
                    if match:
                        boxed_content = match.group(1).strip()
                        break
            
            # Compare boxed content to ground truth
            if boxed_content is not None and i < len(ground_truth):
                ground_truth_item = ground_truth[i]
                if isinstance(ground_truth_item, str):
                    rewards.append(1.0 if boxed_content == ground_truth_item.strip() else self.format_reward)
                elif isinstance(ground_truth_item, (int, float)):
                    # Try to convert boxed content to number for comparison
                    try:
                        boxed_number = float(boxed_content)
                        rewards.append(1.0 if boxed_number == ground_truth_item else self.format_reward)
                    except ValueError:
                        # Boxed content is not a valid number
                        rewards.append(self.format_reward)        
                else:
                    rewards.append(self.format_reward)
            elif boxed_content is not None:
                rewards.append(self.format_reward)
            else:
                rewards.append(0.0)

        return torch.tensor(rewards)




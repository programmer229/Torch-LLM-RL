

import torch
from typing import Protocol, List

from AgentOrchestration.chat.message import Rollout

def length_penalty(rollouts: List[Rollout], ground_truth: List[str]) -> torch.TensorType:
    model_messages = rollouts[-1]
    rewards = torch.tensor([abs(len(model_messages.content)-20)])
    return rewards
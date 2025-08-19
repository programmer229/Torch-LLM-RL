
from torch import TensorType
import torch
from typing import Protocol, List

from AgentOrchestration.chat.message import Rollout

class Reward(Protocol):

    def __call__(self, rollouts: List[Rollout], ground_truth: List[str]) -> TensorType:
        pass
    
        
       
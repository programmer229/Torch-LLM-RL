

from abc import ABC, abstractmethod
from typing import List

from AgentOrchestration.utils.message import Rollouts

class Trainer(ABC):



    def __init__(self) -> None:
        super().__init__()
    

    
    def calculate_loss(self, rollouts: List[Rollouts], agent_model, ref_model, kl_model):
        pass

    
    
    






from abc import ABC, abstractmethod
from typing import List

from SimpleTorchLLMRL.chat.message import Rollout

class Trainer(ABC):



    def __init__(self) -> None:
        super().__init__()
    

    @abstractmethod
    def calculate_loss(self, rollouts: List[Rollout], agent_model, ref_model, kl_model):
            pass

    
    
    




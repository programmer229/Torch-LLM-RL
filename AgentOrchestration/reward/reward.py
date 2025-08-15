

from typing import Protocol, List

from AgentOrchestration.utils.message import Rollout

class Reward(Protocol):

    def __call__(self, rollouts: List[Rollout], ground_truth: List[str]) -> List[float]:
        pass
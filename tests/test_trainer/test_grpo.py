
import torch

from AgentOrchestration.trainer.GRPO import GRPO








def test_grpo_advantage_all_ones():
    grpo = GRPO(None,None,None)
    n = 10
    rewards = torch.ones(n)
    advantage = grpo._calculate_advantage(rewards)
    assert torch.allclose(advantage,torch.zeros(n))



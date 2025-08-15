


from AgentOrchestration.reward.boxed import BoxedReward
from AgentOrchestration.utils.message import Message, Rollout, MessageType


def test_no_box():
    message = Message("ajklsdas", MessageType.MODEL)
    rollout = Rollout()
    rollout.add_message(message)
    reward_func = BoxedReward(format_reward=0.1)
    reward = reward_func(rollouts=[rollout], ground_truth=[2])
    assert reward == [0.0]

def test_box_wrong_solution():
    message = Message("\\boxed{test}", MessageType.MODEL)
    rollout = Rollout()
    rollout.add_message(message)
    reward_func = BoxedReward(format_reward=0.1)
    reward = reward_func(rollouts=[rollout], ground_truth=[2])
    assert reward == [0.1]

def test_box_solution_str():
    message = Message("\\boxed{test}", MessageType.MODEL)
    rollout = Rollout()
    rollout.add_message(message)
    reward_func = BoxedReward(format_reward=0.1)
    reward = reward_func(rollouts=[rollout], ground_truth=["test"])
    assert reward == [1.0]


def test_box_solution_numeric():
    message = Message("\\boxed{1.234}", MessageType.MODEL)
    rollout = Rollout()
    rollout.add_message(message)
    reward_func = BoxedReward(format_reward=0.1)
    reward = reward_func(rollouts=[rollout], ground_truth=[1.234])
    assert reward == [1.0]

def test_box_solution_numeric_sig_figs():
    message = Message("\\boxed{1.0}", MessageType.MODEL)
    rollout = Rollout()
    rollout.add_message(message)
    reward_func = BoxedReward(format_reward=0.1)
    reward = reward_func(rollouts=[rollout], ground_truth=[1])
    assert reward == [1.0]
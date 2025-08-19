"""
Unit tests for AgentOrchestration.reward.boxed module.

Tests cover BoxedReward class functionality including reward calculation,
boxed notation parsing, and different data type comparisons.
"""

import pytest
import torch
from typing import List

from AgentOrchestration.chat.message import Message, MessageType, Rollout
from AgentOrchestration.reward.boxed import BoxedReward


class TestBoxedReward:
    """Test BoxedReward class functionality."""
    
    def test_init_default(self):
        """Test BoxedReward initialization with default format_reward."""
        reward = BoxedReward()
        assert reward.format_reward == 0
    
    def test_init_custom_format_reward(self):
        """Test BoxedReward initialization with custom format_reward."""
        custom_reward = -0.5
        reward = BoxedReward(format_reward=custom_reward)
        assert reward.format_reward == custom_reward
    
    def test_call_single_rollout_correct_answer(self):
        """Test reward calculation for single rollout with correct answer."""
        reward = BoxedReward()
        rollout = Rollout()
        rollout.add_messages(Message("Solve: 2+2", MessageType.MESSAGE))
        rollout.add_messages(Message("The answer is \\boxed{4}", MessageType.MODEL))
        
        rewards = reward([rollout], ["4"])
        
        assert torch.equal(rewards, torch.tensor([1.0]))
    
    def test_call_single_rollout_wrong_answer(self):
        """Test reward calculation for single rollout with wrong answer."""
        reward = BoxedReward(format_reward=-0.1)
        rollout = Rollout()
        rollout.add_messages(Message("Solve: 2+2", MessageType.MESSAGE))
        rollout.add_messages(Message("The answer is \\boxed{5}", MessageType.MODEL))
        
        rewards = reward([rollout], ["4"])
        
        assert torch.equal(rewards, torch.tensor([-0.1]))
    
    def test_call_multiple_rollouts_mixed_results(self):
        """Test reward calculation for multiple rollouts with mixed results."""
        reward = BoxedReward(format_reward=0.0)
        
        rollouts = []
        # Correct answer
        rollout1 = Rollout()
        rollout1.add_messages(Message("What is 10/2?", MessageType.MESSAGE))
        rollout1.add_messages(Message("The answer is \\boxed{5}", MessageType.MODEL))
        rollouts.append(rollout1)
        
        # Wrong answer
        rollout2 = Rollout()
        rollout2.add_messages(Message("What is 3*3?", MessageType.MESSAGE))
        rollout2.add_messages(Message("The answer is \\boxed{10}", MessageType.MODEL))
        rollouts.append(rollout2)
        
        # Correct answer
        rollout3 = Rollout()
        rollout3.add_messages(Message("What is 7-3?", MessageType.MESSAGE))
        rollout3.add_messages(Message("I think it's \\boxed{4}", MessageType.MODEL))
        rollouts.append(rollout3)
        
        ground_truth = ["5", "9", "4"]
        rewards = reward(rollouts, ground_truth)
        
        expected = torch.tensor([1.0, 0.0, 1.0])
        assert torch.equal(rewards, expected)
    
    def test_call_no_boxed_content(self):
        """Test reward calculation when no boxed notation found."""
        reward = BoxedReward()
        rollout = Rollout()
        rollout.add_messages(Message("Solve: 2+2", MessageType.MESSAGE))
        rollout.add_messages(Message("The answer is 4", MessageType.MODEL))  # No \\boxed{}
        
        rewards = reward([rollout], ["4"])
        
        assert torch.equal(rewards, torch.tensor([0.0]))
    
    def test_call_empty_boxed_content(self):
        """Test reward calculation with empty boxed notation."""
        reward = BoxedReward(format_reward=-0.2)
        rollout = Rollout()
        rollout.add_messages(Message("Solve: 2+2", MessageType.MESSAGE))
        rollout.add_messages(Message("The answer is \\boxed{}", MessageType.MODEL))
        
        rewards = reward([rollout], ["4"])
        
        assert torch.equal(rewards, torch.tensor([-0.2]))
    
    def test_call_numeric_ground_truth_correct(self):
        """Test reward calculation with numeric ground truth - correct."""
        reward = BoxedReward()
        rollout = Rollout()
        rollout.add_messages(Message("Calculate: 15/3", MessageType.MESSAGE))
        rollout.add_messages(Message("The result is \\boxed{5.0}", MessageType.MODEL))
        
        rewards = reward([rollout], [5.0])
        
        assert torch.equal(rewards, torch.tensor([1.0]))
    
    def test_call_numeric_ground_truth_wrong(self):
        """Test reward calculation with numeric ground truth - wrong."""
        reward = BoxedReward(format_reward=-0.3)
        rollout = Rollout()
        rollout.add_messages(Message("Calculate: 8*7", MessageType.MESSAGE))
        rollout.add_messages(Message("The result is \\boxed{54}", MessageType.MODEL))
        
        rewards = reward([rollout], [56])
        
        assert torch.equal(rewards, torch.tensor([-0.3]))
    
    def test_call_invalid_numeric_conversion(self):
        """Test reward calculation when boxed content can't be converted to number."""
        reward = BoxedReward(format_reward=-0.1)
        rollout = Rollout()
        rollout.add_messages(Message("Calculate: 2+2", MessageType.MESSAGE))
        rollout.add_messages(Message("The answer is \\boxed{abc}", MessageType.MODEL))
        
        rewards = reward([rollout], [4])
        
        assert torch.equal(rewards, torch.tensor([-0.1]))
    
    def test_call_multiple_boxed_in_message(self):
        """Test reward calculation when message contains multiple boxed notations."""
        reward = BoxedReward()
        rollout = Rollout()
        rollout.add_messages(Message("Solve both: 2+2 and 3+3", MessageType.MESSAGE))
        rollout.add_messages(Message("First \\boxed{4} and second \\boxed{6}", MessageType.MODEL))
        
        rewards = reward([rollout], ["4"])  # Should match first boxed content
        
        assert torch.equal(rewards, torch.tensor([1.0]))
    
    def test_call_boxed_with_whitespace(self):
        """Test reward calculation with whitespace in boxed content."""
        reward = BoxedReward()
        rollout = Rollout()
        rollout.add_messages(Message("What is 5+5?", MessageType.MESSAGE))
        rollout.add_messages(Message("The answer is \\boxed{ 10 }", MessageType.MODEL))
        
        rewards = reward([rollout], [" 10 "])  # Ground truth also has whitespace
        
        assert torch.equal(rewards, torch.tensor([1.0]))
    
    def test_call_boxed_whitespace_mismatch(self):
        """Test reward calculation with whitespace mismatch."""
        reward = BoxedReward(format_reward=0.0)
        rollout = Rollout()
        rollout.add_messages(Message("What is 5+5?", MessageType.MESSAGE))
        rollout.add_messages(Message("The answer is \\boxed{ 10 }", MessageType.MODEL))
        
        rewards = reward([rollout], ["10"])  # Ground truth without whitespace
        
        assert torch.equal(rewards, torch.tensor([0.0]))
    
    def test_call_no_model_messages(self):
        """Test reward calculation when rollout has no MODEL messages."""
        reward = BoxedReward()
        rollout = Rollout()
        rollout.add_messages(Message("System prompt", MessageType.SYSTEM))
        rollout.add_messages(Message("User question", MessageType.MESSAGE))
        
        rewards = reward([rollout], ["42"])
        
        assert torch.equal(rewards, torch.tensor([0.0]))
    
    def test_call_multiple_model_messages(self):
        """Test reward calculation with multiple MODEL messages (should use first with boxed)."""
        reward = BoxedReward()
        rollout = Rollout()
        rollout.add_messages(Message("What is 6*7?", MessageType.MESSAGE))
        rollout.add_messages(Message("Let me think...", MessageType.MODEL))
        rollout.add_messages(Message("The answer is \\boxed{42}", MessageType.MODEL))
        rollout.add_messages(Message("I'm confident in this answer", MessageType.MODEL))
        
        rewards = reward([rollout], ["42"])
        
        assert torch.equal(rewards, torch.tensor([1.0]))
    
    def test_call_mismatched_lengths(self):
        """Test reward calculation with mismatched rollout and ground truth lengths."""
        reward = BoxedReward(format_reward=-0.1)
        
        rollouts = []
        rollout1 = Rollout()
        rollout1.add_messages(Message("The answer is \\boxed{42}", MessageType.MODEL))
        rollouts.append(rollout1)
        
        rollout2 = Rollout()
        rollout2.add_messages(Message("The answer is \\boxed{24}", MessageType.MODEL))
        rollouts.append(rollout2)
        
        # Only one ground truth for two rollouts
        ground_truth = ["42"]
        rewards = reward(rollouts, ground_truth)
        
        # First rollout should match, second should get format_reward
        expected = torch.tensor([1.0, -0.1])
        assert torch.equal(rewards, expected)
    
    def test_call_complex_boxed_content(self):
        """Test reward calculation with complex boxed content."""
        reward = BoxedReward()
        rollout = Rollout()
        rollout.add_messages(Message("Solve the equation", MessageType.MESSAGE))
        rollout.add_messages(Message("The solution is \\boxed{x = 2.5}", MessageType.MODEL))
        
        rewards = reward([rollout], ["x = 2.5"])
        
        assert torch.equal(rewards, torch.tensor([1.0]))
    
    def test_call_nested_braces_in_boxed(self):
        """Test reward calculation with nested braces (should only match outermost)."""
        reward = BoxedReward()
        rollout = Rollout()
        rollout.add_messages(Message("Solve", MessageType.MESSAGE))
        rollout.add_messages(Message("Answer: \\boxed{set = {1, 2, 3}}", MessageType.MODEL))
        
        rewards = reward([rollout], ["set = {1, 2, 3"])  # Note: missing closing brace
        
        assert torch.equal(rewards, torch.tensor([0.0]))
    
    @pytest.mark.parametrize("boxed_content,ground_truth,expected_reward", [
        ("42", "42", 1.0),
        ("42", "43", 0.0),
        ("3.14", "3.14", 1.0),
        ("", "", 1.0),
        ("hello world", "hello world", 1.0),
        ("Hello", "hello", 0.0),  # Case sensitive
    ])
    def test_call_parametrized_string_comparisons(self, boxed_content, ground_truth, expected_reward):
        """Test reward calculation with various string comparisons."""
        reward = BoxedReward(format_reward=0.0)
        rollout = Rollout()
        rollout.add_messages(Message("Question", MessageType.MESSAGE))
        rollout.add_messages(Message(f"Answer: \\boxed{{{boxed_content}}}", MessageType.MODEL))
        
        rewards = reward([rollout], [ground_truth])
        
        assert torch.equal(rewards, torch.tensor([expected_reward]))
    
    @pytest.mark.parametrize("boxed_content,ground_truth,expected_reward", [
        ("42", 42, 1.0),
        ("42.0", 42.0, 1.0),
        ("3.14159", 3.14159, 1.0),
        ("0", 0, 1.0),
        ("-5", -5, 1.0),
        ("42", 43, 0.0),
        ("invalid", 42, 0.0),
    ])
    def test_call_parametrized_numeric_comparisons(self, boxed_content, ground_truth, expected_reward):
        """Test reward calculation with various numeric comparisons."""
        reward = BoxedReward(format_reward=0.0)
        rollout = Rollout()
        rollout.add_messages(Message("Question", MessageType.MESSAGE))
        rollout.add_messages(Message(f"Answer: \\boxed{{{boxed_content}}}", MessageType.MODEL))
        
        rewards = reward([rollout], [ground_truth])
        
        assert torch.equal(rewards, torch.tensor([expected_reward]))
    
    def test_call_return_type(self):
        """Test that reward calculation returns proper torch tensor."""
        reward = BoxedReward()
        rollout = Rollout()
        rollout.add_messages(Message("The answer is \\boxed{42}", MessageType.MODEL))
        
        rewards = reward([rollout], ["42"])
        
        assert isinstance(rewards, torch.Tensor)
        assert rewards.dtype == torch.float32
        assert rewards.shape == (1,)
"""
Global pytest configuration and shared fixtures.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from SimpleTorchLLMRL.chat.message import Message, MessageType, Rollout
from SimpleTorchLLMRL.reward.boxed import BoxedReward


@pytest.fixture
def mock_tokenizer():
    """Mock GPT2 tokenizer for testing."""
    tokenizer = Mock(spec=GPT2Tokenizer)
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test response"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "<|endoftext|>"
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock GPT2 model for testing."""
    model = Mock(spec=GPT2LMHeadModel)
    mock_output = Mock()
    mock_output.logits = torch.tensor([[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]])
    model.return_value = mock_output
    model.parameters.return_value = [torch.tensor([1.0])]
    return model


@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return Message(content="Test message", type=MessageType.MESSAGE)


@pytest.fixture
def sample_rollout():
    """Create a sample rollout with messages."""
    rollout = Rollout()
    rollout.add_messages(Message("System prompt", MessageType.SYSTEM))
    rollout.add_messages(Message("User question", MessageType.MESSAGE))
    rollout.add_messages(Message("Model response with \\boxed{42}", MessageType.MODEL))
    return rollout


@pytest.fixture
def sample_rollouts():
    """Create multiple sample rollouts for batch testing."""
    rollouts = []
    for i in range(3):
        rollout = Rollout()
        rollout.add_messages(Message(f"System prompt {i}", MessageType.SYSTEM))
        rollout.add_messages(Message(f"Question {i}", MessageType.MESSAGE))
        rollout.add_messages(Message(f"Answer \\boxed{{{i * 10}}}", MessageType.MODEL))
        rollouts.append(rollout)
    return rollouts


@pytest.fixture
def boxed_reward():
    """Create a BoxedReward instance."""
    return BoxedReward(format_reward=0.0)


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    dataset = Mock()
    dataset.__getitem__ = Mock(return_value={"text": "Sample question", "label": "42"})
    dataset.__len__ = Mock(return_value=100)
    return dataset


@pytest.fixture
def torch_device():
    """Get appropriate torch device for testing."""
    return torch.device("cpu")
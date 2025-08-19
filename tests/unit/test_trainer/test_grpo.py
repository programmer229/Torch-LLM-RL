"""
Unit tests for AgentOrchestration.trainer.GRPO module.

Tests cover GRPO trainer functionality including advantage calculation,
probability ratios, loss calculation, and training integration.
"""

import pytest
import torch
import copy
from unittest.mock import Mock, MagicMock, patch
from typing import List

from AgentOrchestration.trainer.GRPO import GRPO
from AgentOrchestration.chat.message import Message, MessageType, Rollout


class TestGRPO:
    """Test GRPO trainer class functionality."""
    
    def test_init(self, mock_model, mock_tokenizer):
        """Test GRPO initialization."""
        eps = 0.2
        trainer = GRPO(model=mock_model, tokenizer=mock_tokenizer, eps=eps)
        
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.eps == eps
        assert trainer.ref_model is not None
    
    def test_init_no_model(self, mock_tokenizer):
        """Test GRPO initialization with None model."""
        trainer = GRPO(model=None, tokenizer=mock_tokenizer, eps=0.1)
        
        assert trainer.model is None
        assert trainer.ref_model is None
    
    def test_calculate_advantage_valid_rewards(self):
        """Test advantage calculation with valid reward tensor."""
        trainer = GRPO(model=Mock(), tokenizer=Mock(), eps=0.1)
        rewards = torch.tensor([1.0, 0.5, 0.0, 1.5])
        
        advantage = trainer._calculate_advantage(rewards)
        
        # Check that advantage has correct shape and properties
        assert advantage.shape == rewards.shape
        assert torch.isfinite(advantage).all()
        
        # Advantage should be normalized (mean ≈ 0, std ≈ 1)
        assert abs(advantage.mean().item()) < 0.01
        assert abs(advantage.std().item() - 1.0) < 0.01
    
    def test_calculate_advantage_single_reward(self):
        """Test advantage calculation with insufficient rewards."""
        trainer = GRPO(model=Mock(), tokenizer=Mock(), eps=0.1)
        rewards = torch.tensor([1.0])
        
        with pytest.raises(ValueError, match="Rollouts much be more than 1"):
            trainer._calculate_advantage(rewards)
    
    def test_calculate_advantage_empty_rewards(self):
        """Test advantage calculation with empty rewards."""
        trainer = GRPO(model=Mock(), tokenizer=Mock(), eps=0.1)
        rewards = torch.tensor([])
        
        with pytest.raises(ValueError, match="Rollouts much be more than 1"):
            trainer._calculate_advantage(rewards)
    
    def test_calculate_advantage_identical_rewards(self):
        """Test advantage calculation with identical rewards (zero std)."""
        trainer = GRPO(model=Mock(), tokenizer=Mock(), eps=0.1)
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        
        advantage = trainer._calculate_advantage(rewards)
        
        # With zero std, advantage should be all zeros (handled by nan_to_num)
        assert torch.allclose(advantage, torch.zeros_like(advantage))
    
    def test_calculate_prob_ratios(self, mock_model, mock_tokenizer):
        """Test probability ratio calculation."""
        # Setup mock model outputs
        mock_logits = torch.tensor([[[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]])
        mock_output = Mock()
        mock_output.logits = mock_logits
        
        mock_model.return_value = mock_output
        
        trainer = GRPO(model=mock_model, tokenizer=mock_tokenizer, eps=0.1)
        trainer.ref_model = Mock()
        trainer.ref_model.return_value = mock_output  # Same output for simplicity
        
        input_ids = torch.tensor([[1, 2, 3]])
        ratios = trainer._calculate_prob_ratios(input_ids)
        
        # With identical model and ref_model outputs, ratios should be 1
        assert ratios.shape == mock_logits.shape
        assert torch.allclose(ratios, torch.ones_like(ratios))
    
    def test_calculate_prob_ratios_different_models(self, mock_tokenizer):
        """Test probability ratio calculation with different model outputs."""
        # Create models with different outputs
        model = Mock()
        ref_model = Mock()
        
        model_output = Mock()
        model_output.logits = torch.tensor([[[2.0, 1.0, 0.0]]])
        model.return_value = model_output
        
        ref_output = Mock()
        ref_output.logits = torch.tensor([[[1.0, 1.0, 1.0]]])
        ref_model.return_value = ref_output
        
        trainer = GRPO(model=model, tokenizer=mock_tokenizer, eps=0.1)
        trainer.ref_model = ref_model
        
        input_ids = torch.tensor([[1, 2, 3]])
        ratios = trainer._calculate_prob_ratios(input_ids)
        
        # Check that ratios have correct shape
        assert ratios.shape == (1, 1, 3)
        assert torch.isfinite(ratios).all()
    
    @patch('torch.tensor')
    def test_calculate_loss_basic(self, mock_tensor_call, mock_model, mock_tokenizer):
        """Test basic loss calculation functionality."""
        # Setup trainer
        trainer = GRPO(model=mock_model, tokenizer=mock_tokenizer, eps=0.2)
        trainer.ref_model = mock_model
        
        # Mock tokenizer behavior
        mock_tokenizer.encode.side_effect = [[1, 2, 3], [4, 5, 6]]
        mock_tokenizer.pad_token_id = 0
        
        # Create sample rollouts
        rollouts = []
        for i in range(2):
            rollout = Rollout()
            rollout.add_messages(Message(f"Question {i}", MessageType.MESSAGE))
            rollout.add_messages(Message(f"Answer {i}", MessageType.MODEL))
            rollouts.append(rollout)
        
        rewards = torch.tensor([1.0, 0.0])
        
        # Mock torch.tensor calls to return predictable padded sequences
        mock_tensor_call.side_effect = [
            torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 0, 0, 0]]),  # inputs_ids
            torch.tensor([])  # Empty tensor for advantage calculation
        ]
        
        # Mock model outputs for probability calculation
        mock_logits = torch.tensor([[[0.1, 0.2, 0.7] for _ in range(6)] for _ in range(2)])
        mock_output = Mock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output
        
        # This should not raise an error
        loss = trainer.calculate_loss(rollouts, rewards)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad or loss.item() == 0.0
    
    def test_calculate_loss_empty_rollouts(self, mock_model, mock_tokenizer):
        """Test loss calculation with empty rollouts."""
        trainer = GRPO(model=mock_model, tokenizer=mock_tokenizer, eps=0.1)
        
        rollouts = []
        rewards = torch.tensor([])
        
        loss = trainer.calculate_loss(rollouts, rewards)
        
        assert torch.equal(loss, torch.tensor(0.0))
    
    def test_calculate_loss_no_model_messages(self, mock_model, mock_tokenizer):
        """Test loss calculation with rollouts containing no MODEL messages."""
        trainer = GRPO(model=mock_model, tokenizer=mock_tokenizer, eps=0.1)
        trainer.ref_model = mock_model
        
        # Create rollouts with no MODEL messages
        rollouts = []
        rollout = Rollout()
        rollout.add_messages(Message("System prompt", MessageType.SYSTEM))
        rollout.add_messages(Message("User question", MessageType.MESSAGE))
        rollouts.append(rollout)
        
        rewards = torch.tensor([1.0])
        
        # Mock tokenizer
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.pad_token_id = 0
        
        # Mock model output
        mock_output = Mock()
        mock_output.logits = torch.tensor([[[0.3, 0.3, 0.4]]])
        mock_model.return_value = mock_output
        
        loss = trainer.calculate_loss(rollouts, rewards)
        
        # Should return zero loss when no MODEL messages to train on
        assert torch.equal(loss, torch.tensor(0.0))
    
    def test_calculate_loss_single_rollout_insufficient_rewards(self, mock_model, mock_tokenizer):
        """Test loss calculation with single rollout (insufficient for advantage)."""
        trainer = GRPO(model=mock_model, tokenizer=mock_tokenizer, eps=0.1)
        
        rollout = Rollout()
        rollout.add_messages(Message("Question", MessageType.MESSAGE))
        rollout.add_messages(Message("Answer", MessageType.MODEL))
        
        rollouts = [rollout]
        rewards = torch.tensor([1.0])
        
        with pytest.raises(ValueError, match="Rollouts much be more than 1"):
            trainer.calculate_loss(rollouts, rewards)
    
    @pytest.mark.parametrize("eps", [0.1, 0.2, 0.3])
    def test_calculate_loss_different_eps_values(self, eps, mock_model, mock_tokenizer):
        """Test loss calculation with different epsilon values."""
        trainer = GRPO(model=mock_model, tokenizer=mock_tokenizer, eps=eps)
        trainer.ref_model = mock_model
        
        # Create minimal rollouts for testing
        rollouts = []
        for i in range(2):
            rollout = Rollout()
            rollout.add_messages(Message(f"Q{i}", MessageType.MESSAGE))
            rollout.add_messages(Message(f"A{i}", MessageType.MODEL))
            rollouts.append(rollout)
        
        rewards = torch.tensor([1.0, 0.0])
        
        # Mock tokenizer and model
        mock_tokenizer.encode.side_effect = [[1, 2], [3, 4]]
        mock_tokenizer.pad_token_id = 0
        
        mock_output = Mock()
        mock_output.logits = torch.tensor([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
        mock_model.return_value = mock_output
        
        # Should not raise error and should use the specified epsilon
        loss = trainer.calculate_loss(rollouts, rewards)
        assert isinstance(loss, torch.Tensor)
        assert trainer.eps == eps
    
    def test_inheritance_from_trainer(self):
        """Test that GRPO inherits from Trainer base class."""
        trainer = GRPO(model=Mock(), tokenizer=Mock(), eps=0.1)
        
        # Check that it has the expected structure
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'tokenizer')
        assert hasattr(trainer, 'eps')
        assert hasattr(trainer, 'calculate_loss')
    
    def test_ref_model_deepcopy(self, mock_model, mock_tokenizer):
        """Test that reference model is a deep copy of the main model."""
        trainer = GRPO(model=mock_model, tokenizer=mock_tokenizer, eps=0.1)
        
        # ref_model should be a different object but with same structure
        assert trainer.ref_model is not trainer.model
        assert trainer.ref_model is not None
    
    @pytest.mark.slow
    def test_calculate_loss_realistic_scenario(self, mock_tokenizer):
        """Test loss calculation with more realistic model and data."""
        # Use actual small tensors for more realistic testing
        model = Mock()
        ref_model = Mock()
        
        # Create realistic logits
        batch_size, seq_len, vocab_size = 2, 4, 100
        model_logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
        ref_logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
        
        model_output = Mock()
        model_output.logits = model_logits
        model.return_value = model_output
        
        ref_output = Mock()
        ref_output.logits = ref_logits
        ref_model.return_value = ref_output
        
        trainer = GRPO(model=model, tokenizer=mock_tokenizer, eps=0.2)
        trainer.ref_model = ref_model
        
        # Create realistic rollouts
        rollouts = []
        for i in range(batch_size):
            rollout = Rollout()
            rollout.add_messages(Message(f"Solve problem {i}", MessageType.MESSAGE))
            rollout.add_messages(Message(f"Solution {i}", MessageType.MODEL))
            rollouts.append(rollout)
        
        rewards = torch.tensor([1.0, 0.5])
        
        # Mock tokenizer to return sequences of appropriate length
        mock_tokenizer.encode.side_effect = [[i] * seq_len for i in range(1, batch_size + 1)]
        mock_tokenizer.pad_token_id = 0
        
        loss = trainer.calculate_loss(rollouts, rewards)
        
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)
        assert loss.numel() == 1  # Scalar loss
"""
Unit tests for SimpleTorchLLMRL.model.generate module.

Tests cover ModelGenerate class functionality including generation methods,
rollout processing, and batch operations.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from typing import List

from SimpleTorchLLMRL.model.generate import ModelGenerate
from SimpleTorchLLMRL.chat.message import Message, MessageType, Rollout


class TestModelGenerate:
    """Test ModelGenerate class functionality."""
    
    def test_init(self, mock_model, mock_tokenizer):
        """Test ModelGenerate initialization."""
        generator = ModelGenerate(model=mock_model, tokenizer=mock_tokenizer)
        
        assert generator.model == mock_model
        assert generator.tokenizer == mock_tokenizer
        assert generator.max_new_tokens == 50
        assert generator.temperature == 0.7
        assert generator.do_sample is True
    
    def test_generate_method_exists(self, mock_model, mock_tokenizer):
        """Test that _generate method exists and is callable."""
        generator = ModelGenerate(model=mock_model, tokenizer=mock_tokenizer)
        
        assert hasattr(generator, '_generate')
        assert callable(generator._generate)
    
    def test_generate_method_actual_implementation(self, mock_model, mock_tokenizer):
        """Test that _generate method has actual implementation."""
        mock_tokenizer.eos_token_id = 50256
        mock_model.eval.return_value = None
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        
        generator = ModelGenerate(model=mock_model, tokenizer=mock_tokenizer)
        
        input_ids = torch.tensor([[1, 2, 3]])
        result = generator._generate(input_ids)
        
        # Should return the new tokens only
        expected = torch.tensor([[4, 5, 6]])
        assert torch.equal(result, expected)
        mock_model.eval.assert_called_once()
        mock_model.generate.assert_called_once()
    
    def test_rollout_generate_response(self, mock_model, mock_tokenizer):
        """Test single rollout response generation."""
        # Setup mocks
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        mock_tokenizer.decode.return_value = "Generated response"
        
        generator = ModelGenerate(model=mock_model, tokenizer=mock_tokenizer)
        
        # Mock the _generate method to return tensor
        generator._generate = Mock(return_value=torch.tensor([[5, 6, 7]]))
        
        # Create test rollout
        rollout = Rollout()
        rollout.add_messages(Message("System prompt", MessageType.SYSTEM))
        rollout.add_messages(Message("User question", MessageType.MESSAGE))
        
        # Execute
        response = generator.rollout_generate_response(rollout)
        
        # Verify results
        assert response.content == "Generated response"
        assert response.type == MessageType.MODEL
        
        # Verify method calls
        mock_tokenizer.encode.assert_called_once()
        # Just verify decode was called with expected arguments
        assert mock_tokenizer.decode.call_count == 1
        call_args = mock_tokenizer.decode.call_args[0]
        assert torch.equal(call_args[0], torch.tensor([5, 6, 7]))
        assert mock_tokenizer.decode.call_args[1]['skip_special_tokens'] is True
    
    def test_rollout_generate_response_empty_rollout(self, mock_model, mock_tokenizer):
        """Test response generation with empty rollout."""
        mock_tokenizer.encode.return_value = []
        mock_tokenizer.decode.return_value = "Empty response"
        
        generator = ModelGenerate(model=mock_model, tokenizer=mock_tokenizer)
        generator._generate = Mock(return_value=torch.tensor([[1, 2]]))
        
        rollout = Rollout()
        
        response = generator.rollout_generate_response(rollout)
        
        assert response.type == MessageType.MODEL
        assert response.content == "Empty response"
    
    def test_batch_rollout_generate_response_basic(self, mock_model, mock_tokenizer):
        """Test batch rollout response generation - basic functionality."""
        # Mock tokenizer to return appropriate structure
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 0], [3, 4, 0], [5, 6, 0]])}
        mock_tokenizer.decode.side_effect = ["Response 1", "Response 2", "Response 3"]
        
        generator = ModelGenerate(model=mock_model, tokenizer=mock_tokenizer)
        generator._generate = Mock(return_value=torch.tensor([[7, 8], [9, 10], [11, 12]]))
        
        # Create test rollouts
        rollouts = []
        for i in range(3):
            rollout = Rollout()
            rollout.add_messages(Message(f"Question {i}", MessageType.MESSAGE))
            rollouts.append(rollout)
        
        initial_lengths = [len(rollout) for rollout in rollouts]
        
        # Execute
        generator.batch_rollout_generate_response(rollouts)
        
        # Verify all rollouts got responses
        for i, rollout in enumerate(rollouts):
            assert len(rollout) == initial_lengths[i] + 1
            assert rollout[-1].type == MessageType.MODEL
            assert rollout[-1].content == f"Response {i + 1}"
    
    def test_batch_rollout_generate_response_empty_list(self, mock_model, mock_tokenizer):
        """Test batch generation with empty rollouts list."""
        # Mock tokenizer to return appropriate structure for empty case
        mock_tokenizer.return_value = {"input_ids": torch.tensor([]).reshape(0, 0)}
        generator = ModelGenerate(model=mock_model, tokenizer=mock_tokenizer)
        generator._generate = Mock(return_value=torch.tensor([]).reshape(0, 0))
        
        rollouts = []
        
        # Should not raise error with empty list
        generator.batch_rollout_generate_response(rollouts)
    
    def test_constructor_parameters(self):
        """Test that constructor accepts correct parameter names."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Constructor should accept tokenizer parameter correctly
        generator = ModelGenerate(model=mock_model, tokenizer=mock_tokenizer)
        assert generator.tokenizer == mock_tokenizer
        assert generator.model == mock_model
    
    def test_error_handling_encode_failure(self, mock_model, mock_tokenizer):
        """Test error handling when tokenizer encode fails."""
        mock_tokenizer.encode.side_effect = Exception("Encoding failed")
        
        generator = ModelGenerate(model=mock_model, tokenizer=mock_tokenizer)
        rollout = Rollout()
        rollout.add_messages(Message("Test", MessageType.MESSAGE))
        
        with pytest.raises(Exception, match="Encoding failed"):
            generator.rollout_generate_response(rollout)
    
    def test_error_handling_decode_failure(self, mock_model, mock_tokenizer):
        """Test error handling when tokenizer decode fails."""
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.side_effect = Exception("Decoding failed")
        
        generator = ModelGenerate(model=mock_model, tokenizer=mock_tokenizer)
        generator._generate = Mock(return_value=torch.tensor([[4, 5, 6]]))
        
        rollout = Rollout()
        rollout.add_messages(Message("Test", MessageType.MESSAGE))
        
        with pytest.raises(Exception, match="Decoding failed"):
            generator.rollout_generate_response(rollout)
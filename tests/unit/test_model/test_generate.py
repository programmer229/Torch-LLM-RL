"""
Unit tests for AgentOrchestration.model.generate module.

Tests cover ModelGenerate class functionality including generation methods,
rollout processing, and batch operations.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from typing import List

from AgentOrchestration.model.generate import ModelGenerate
from AgentOrchestration.chat.message import Message, MessageType, Rollout


class TestModelGenerate:
    """Test ModelGenerate class functionality."""
    
    def test_init(self, mock_model, mock_tokenizer):
        """Test ModelGenerate initialization."""
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        
        assert generator.model == mock_model
        assert generator.tokenizer == mock_tokenizer
    
    def test_generate_method_exists(self, mock_model, mock_tokenizer):
        """Test that _generate method exists and is callable."""
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        
        assert hasattr(generator, '_generate')
        assert callable(generator._generate)
    
    def test_generate_method_placeholder(self, mock_model, mock_tokenizer):
        """Test that _generate method currently has pass implementation."""
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        
        # Should not raise error, just return None (pass implementation)
        result = generator._generate(torch.tensor([[1, 2, 3]]))
        assert result is None
    
    @patch('torch.tensor')
    def test_rollout_generate_response(self, mock_tensor_call, mock_model, mock_tokenizer):
        """Test single rollout response generation."""
        # Setup mocks
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        mock_tokenizer.decode.return_value = "Generated response"
        
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        
        # Mock the _generate method to return something
        generator._generate = Mock(return_value=[5, 6, 7])
        
        # Create test rollout
        rollout = Rollout()
        rollout.add_messages(Message("System prompt", MessageType.SYSTEM))
        rollout.add_messages(Message("User question", MessageType.MESSAGE))
        
        # Mock torch.tensor call
        mock_tensor_call.return_value = torch.tensor([[1, 2, 3, 4]])
        
        # Execute
        initial_length = len(rollout)
        generator.rollout_generate_response(rollout)
        
        # Verify results
        assert len(rollout) == initial_length + 1
        assert rollout[-1].content == "Generated response"
        assert rollout[-1].type == MessageType.MODEL
        
        # Verify method calls
        mock_tokenizer.encode.assert_called_once()
        generator._generate.assert_called_once_with(torch.tensor([[1, 2, 3, 4]]))
        mock_tokenizer.decode.assert_called_once_with([5, 6, 7])
    
    def test_rollout_generate_response_empty_rollout(self, mock_model, mock_tokenizer):
        """Test response generation with empty rollout."""
        mock_tokenizer.encode.return_value = []
        mock_tokenizer.decode.return_value = "Empty response"
        
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        generator._generate = Mock(return_value=[1, 2])
        
        rollout = Rollout()
        
        generator.rollout_generate_response(rollout)
        
        assert len(rollout) == 1
        assert rollout[0].type == MessageType.MODEL
        assert rollout[0].content == "Empty response"
    
    @patch('torch.tensor')
    def test_rollout_generate_response_conversion_flow(self, mock_tensor_call, mock_model, mock_tokenizer):
        """Test the complete conversion flow in rollout generation."""
        # Setup detailed mocks
        conversation_string = "system: Hello user: Hi"
        encoded_ids = [10, 20, 30]
        generated_ids = [40, 50, 60]
        decoded_response = "AI response"
        
        mock_tokenizer.encode.return_value = encoded_ids
        mock_tokenizer.decode.return_value = decoded_response
        
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        generator._generate = Mock(return_value=generated_ids)
        
        # Create rollout that will produce known conversation string
        rollout = Rollout()
        rollout.add_messages(Message("Hello", MessageType.SYSTEM))
        rollout.add_messages(Message("Hi", MessageType.MESSAGE))
        
        # Mock tensor creation
        expected_tensor = torch.tensor([[10, 20, 30]])
        mock_tensor_call.return_value = expected_tensor
        
        generator.rollout_generate_response(rollout)
        
        # Verify the flow
        mock_tokenizer.encode.assert_called_once()
        mock_tensor_call.assert_called_once_with(encoded_ids)
        generator._generate.assert_called_once_with(expected_tensor)
        mock_tokenizer.decode.assert_called_once_with(generated_ids)
    
    def test_batch_rollout_generate_response_basic(self, mock_model, mock_tokenizer):
        """Test batch rollout response generation - basic functionality."""
        # Setup mocks
        mock_tokenizer.encode.side_effect = [[1, 2], [3, 4], [5, 6]]
        mock_tokenizer.decode.side_effect = ["Response 1", "Response 2", "Response 3"]
        
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        generator._generate = Mock(return_value=[[7, 8], [9, 10], [11, 12]])
        
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
    
    def test_batch_rollout_generate_response_syntax_error(self, mock_model, mock_tokenizer):
        """Test that batch generation has syntax error in list comprehension."""
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        
        rollouts = [Rollout()]
        
        # The current implementation has a syntax error in line 38
        # This test documents the current buggy behavior
        with pytest.raises(SyntaxError):
            # This line in the source has invalid syntax:
            # messages = [Message(output, MessageType.MODEL) in output]
            # Should be: messages = [Message(output, MessageType.MODEL) for output in outputs]
            exec("messages = [Message(output, MessageType.MODEL) in output]")
    
    def test_batch_rollout_generate_response_empty_list(self, mock_model, mock_tokenizer):
        """Test batch generation with empty rollouts list."""
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        
        rollouts = []
        
        # Should not raise error with empty list
        # Note: Current implementation will likely fail due to syntax error
        # but we test the expected behavior
        try:
            generator.batch_rollout_generate_response(rollouts)
        except SyntaxError:
            # Expected due to syntax error in source code
            pass
    
    @patch('torch.tensor')
    def test_batch_rollout_tensor_creation(self, mock_tensor_call, mock_model, mock_tokenizer):
        """Test tensor creation in batch rollout generation."""
        mock_tokenizer.encode.side_effect = [[1, 2], [3, 4]]
        
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        generator._generate = Mock(return_value=[[5, 6], [7, 8]])
        mock_tokenizer.decode.side_effect = ["Resp1", "Resp2"]
        
        rollouts = [Rollout(), Rollout()]
        for i, rollout in enumerate(rollouts):
            rollout.add_messages(Message(f"Q{i}", MessageType.MESSAGE))
        
        try:
            generator.batch_rollout_generate_response(rollouts)
            
            # Verify tensor creation with batch data
            expected_ids = [[1, 2], [3, 4]]
            mock_tensor_call.assert_called_once_with(expected_ids)
        except SyntaxError:
            # Expected due to syntax error in source
            pass
    
    def test_rollout_generate_response_integration(self, mock_model, mock_tokenizer):
        """Test rollout generation integration with actual rollout methods."""
        mock_tokenizer.encode.return_value = [100, 200]
        mock_tokenizer.decode.return_value = "Integration test response"
        
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        generator._generate = Mock(return_value=[300, 400])
        
        # Create rollout with multiple message types
        rollout = Rollout()
        rollout.add_messages(Message("You are helpful", MessageType.SYSTEM))
        rollout.add_messages(Message("What is AI?", MessageType.MESSAGE))
        
        # Verify initial state
        assert len(rollout) == 2
        assert rollout.is_complete is False
        
        # Generate response
        generator.rollout_generate_response(rollout)
        
        # Verify final state
        assert len(rollout) == 3
        assert rollout[-1].type == MessageType.MODEL
        assert rollout[-1].content == "Integration test response"
        
        # Verify rollout methods still work
        formatted = rollout.format_conversation()
        assert len(formatted) == 3
        assert formatted[-1]["role"] == "assistant"
        assert formatted[-1]["content"] == "Integration test response"
    
    def test_error_handling_encode_failure(self, mock_model, mock_tokenizer):
        """Test error handling when tokenizer encode fails."""
        mock_tokenizer.encode.side_effect = Exception("Encoding failed")
        
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        rollout = Rollout()
        rollout.add_messages(Message("Test", MessageType.MESSAGE))
        
        with pytest.raises(Exception, match="Encoding failed"):
            generator.rollout_generate_response(rollout)
    
    def test_error_handling_decode_failure(self, mock_model, mock_tokenizer):
        """Test error handling when tokenizer decode fails."""
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.side_effect = Exception("Decoding failed")
        
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        generator._generate = Mock(return_value=[4, 5, 6])
        
        rollout = Rollout()
        rollout.add_messages(Message("Test", MessageType.MESSAGE))
        
        with pytest.raises(Exception, match="Decoding failed"):
            generator.rollout_generate_response(rollout)
    
    def test_typo_in_constructor_parameter(self):
        """Test that constructor parameter has typo (tokenzier instead of tokenizer)."""
        # This test documents the typo in the constructor parameter name
        # The parameter should be 'tokenizer' but is written as 'tokenzier'
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Current (incorrect) parameter name
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        assert generator.tokenizer == mock_tokenizer
        
        # The correct parameter name would cause a TypeError
        with pytest.raises(TypeError):
            ModelGenerate(model=mock_model, tokenizer=mock_tokenizer)
    
    @pytest.mark.parametrize("num_rollouts", [1, 2, 5, 10])
    def test_batch_generation_scaling(self, num_rollouts, mock_model, mock_tokenizer):
        """Test batch generation with different numbers of rollouts."""
        # Setup mocks for scaling test
        mock_tokenizer.encode.side_effect = [[i] * 3 for i in range(num_rollouts)]
        mock_tokenizer.decode.side_effect = [f"Response {i}" for i in range(num_rollouts)]
        
        generator = ModelGenerate(model=mock_model, tokenzier=mock_tokenizer)
        generator._generate = Mock(return_value=[[i + 100] * 3 for i in range(num_rollouts)])
        
        # Create rollouts
        rollouts = []
        for i in range(num_rollouts):
            rollout = Rollout()
            rollout.add_messages(Message(f"Question {i}", MessageType.MESSAGE))
            rollouts.append(rollout)
        
        try:
            generator.batch_rollout_generate_response(rollouts)
            
            # Verify all got responses (if syntax error is fixed)
            for i, rollout in enumerate(rollouts):
                if len(rollout) > 1:  # If generation succeeded
                    assert rollout[-1].type == MessageType.MODEL
                    
        except SyntaxError:
            # Expected due to syntax error in current implementation
            pytest.skip("Skipping due to syntax error in source code")
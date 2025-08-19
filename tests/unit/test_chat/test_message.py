"""
Unit tests for AgentOrchestration.chat.message module.

Tests cover Message dataclass, MessageType enum, and Rollout class functionality.
"""

import pytest
from typing import List

from AgentOrchestration.chat.message import Message, MessageType, Rollout


class TestMessageType:
    """Test MessageType enum values and behavior."""
    
    def test_message_type_values(self):
        """Test that MessageType enum has correct values."""
        assert MessageType.SYSTEM.value == "system"
        assert MessageType.MESSAGE.value == "user"
        assert MessageType.MODEL.value == "assistant"
    
    def test_message_type_members(self):
        """Test that MessageType has expected members."""
        expected_members = {"SYSTEM", "MESSAGE", "MODEL"}
        actual_members = {member.name for member in MessageType}
        assert actual_members == expected_members


class TestMessage:
    """Test Message dataclass functionality."""
    
    def test_message_creation_minimal(self):
        """Test creating a message with required fields only."""
        message = Message(content="Hello", type=MessageType.MESSAGE)
        
        assert message.content == "Hello"
        assert message.type == MessageType.MESSAGE
        assert message.tokenizer_ids is None
    
    def test_message_creation_with_tokenizer_ids(self):
        """Test creating a message with tokenizer IDs."""
        tokenizer_ids = [1, 2, 3, 4, 5]
        message = Message(
            content="Hello world",
            type=MessageType.MODEL,
            tokenizer_ids=tokenizer_ids
        )
        
        assert message.content == "Hello world"
        assert message.type == MessageType.MODEL
        assert message.tokenizer_ids == tokenizer_ids
    
    @pytest.mark.parametrize("message_type", [
        MessageType.SYSTEM,
        MessageType.MESSAGE, 
        MessageType.MODEL
    ])
    def test_message_with_different_types(self, message_type):
        """Test message creation with different MessageType values."""
        message = Message(content="Test content", type=message_type)
        assert message.type == message_type


class TestRollout:
    """Test Rollout class functionality."""
    
    def test_rollout_initialization(self):
        """Test rollout initializes with correct defaults."""
        rollout = Rollout()
        
        assert len(rollout) == 0
        assert rollout.is_complete is False
        assert rollout.reward is None
        assert rollout._messages == []
    
    def test_rollout_add_messages(self, sample_message):
        """Test adding a message to rollout."""
        rollout = Rollout()
        rollout.add_messages(sample_message)
        
        assert len(rollout) == 1
        assert rollout[0] == sample_message
    
    def test_rollout_add_multiple_messages(self):
        """Test adding multiple messages to rollout."""
        rollout = Rollout()
        messages = [
            Message("System", MessageType.SYSTEM),
            Message("User", MessageType.MESSAGE),
            Message("Assistant", MessageType.MODEL)
        ]
        
        for message in messages:
            rollout.add_messages(message)
        
        assert len(rollout) == 3
        for i, message in enumerate(messages):
            assert rollout[i] == message
    
    def test_rollout_indexing(self):
        """Test rollout indexing functionality."""
        rollout = Rollout()
        message1 = Message("First", MessageType.MESSAGE)
        message2 = Message("Second", MessageType.MODEL)
        
        rollout.add_messages(message1)
        rollout.add_messages(message2)
        
        assert rollout[0] == message1
        assert rollout[1] == message2
        assert rollout[-1] == message2
        assert rollout[-2] == message1
    
    def test_rollout_indexing_out_of_bounds(self):
        """Test rollout indexing with invalid indices."""
        rollout = Rollout()
        rollout.add_messages(Message("Test", MessageType.MESSAGE))
        
        with pytest.raises(IndexError):
            _ = rollout[5]
        
        with pytest.raises(IndexError):
            _ = rollout[-5]
    
    def test_rollout_len(self):
        """Test rollout length calculation."""
        rollout = Rollout()
        assert len(rollout) == 0
        
        rollout.add_messages(Message("Test1", MessageType.MESSAGE))
        assert len(rollout) == 1
        
        rollout.add_messages(Message("Test2", MessageType.MODEL))
        assert len(rollout) == 2
    
    def test_format_conversation(self):
        """Test conversation formatting for API compatibility."""
        rollout = Rollout()
        rollout.add_messages(Message("You are helpful", MessageType.SYSTEM))
        rollout.add_messages(Message("What is 2+2?", MessageType.MESSAGE))
        rollout.add_messages(Message("2+2 equals 4", MessageType.MODEL))
        
        formatted = rollout.format_conversation()
        expected = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4"}
        ]
        
        assert formatted == expected
    
    def test_format_conversation_empty(self):
        """Test formatting empty conversation."""
        rollout = Rollout()
        formatted = rollout.format_conversation()
        assert formatted == []
    
    def test_format_conversation_str(self):
        """Test string formatting of conversation."""
        rollout = Rollout()
        rollout.add_messages(Message("System prompt", MessageType.SYSTEM))
        rollout.add_messages(Message("User query", MessageType.MESSAGE))
        rollout.add_messages(Message("Model response", MessageType.MODEL))
        
        formatted_str = rollout.format_conversation_str()
        expected = "system: System promptuser: User queryassistant: Model response"
        
        assert formatted_str == expected
    
    def test_format_conversation_str_empty(self):
        """Test string formatting of empty conversation."""
        rollout = Rollout()
        formatted_str = rollout.format_conversation_str()
        assert formatted_str == ""
    
    def test_rollout_state_management(self):
        """Test rollout completion and reward state."""
        rollout = Rollout()
        
        # Initially not complete
        assert rollout.is_complete is False
        assert rollout.reward is None
        
        # Set completion state
        rollout.is_complete = True
        rollout.reward = 0.85
        
        assert rollout.is_complete is True
        assert rollout.reward == 0.85
    
    @pytest.mark.parametrize("content,message_type", [
        ("System message", MessageType.SYSTEM),
        ("User message", MessageType.MESSAGE),
        ("Model message", MessageType.MODEL),
        ("", MessageType.MESSAGE),  # Empty content
        ("Multi\nline\ncontent", MessageType.MODEL),  # Multi-line content
    ])
    def test_rollout_with_various_messages(self, content, message_type):
        """Test rollout with various message types and content."""
        rollout = Rollout()
        message = Message(content=content, type=message_type)
        rollout.add_messages(message)
        
        assert len(rollout) == 1
        assert rollout[0].content == content
        assert rollout[0].type == message_type
    
    def test_rollout_iteration(self):
        """Test iterating over rollout messages."""
        rollout = Rollout()
        messages = [
            Message("Msg1", MessageType.SYSTEM),
            Message("Msg2", MessageType.MESSAGE),
            Message("Msg3", MessageType.MODEL)
        ]
        
        for msg in messages:
            rollout.add_messages(msg)
        
        # Test iteration
        iterated_messages = list(rollout)
        assert iterated_messages == messages
        
        # Test with enumerate
        for i, msg in enumerate(rollout):
            assert msg == messages[i]
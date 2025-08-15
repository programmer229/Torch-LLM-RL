import pytest
from AgentOrchestration.utils.message import Message, MessageType, Rollout


class TestMessageType:
    """Test the MessageType enum."""
    
    def test_message_type_values(self):
        """Test that MessageType enum has expected values."""
        assert MessageType.SYSTEM.value == "system"
        assert MessageType.PROMPT.value == "prompt"
        assert MessageType.MESSAGE.value == "user"
        assert MessageType.MODEL.value == "assistant"
    
    def test_message_type_members(self):
        """Test that all expected MessageType members exist."""
        expected_types = {"SYSTEM", "PROMPT", "MESSAGE", "MODEL"}
        actual_types = {member.name for member in MessageType}
        assert actual_types == expected_types


class TestMessage:
    """Test the Message dataclass."""
    
    def test_message_creation_basic(self):
        """Test basic message creation."""
        content = "Hello, world!"
        msg_type = MessageType.SYSTEM
        message = Message(content, msg_type)
        
        assert message.content == content
        assert message.type == msg_type
        assert message.tokenizer_ids is None
    
    def test_message_creation_with_tokenizer_ids(self):
        """Test message creation with tokenizer IDs."""
        content = "Test message"
        msg_type = MessageType.MODEL
        tokenizer_ids = [1, 2, 3, 4, 5]
        message = Message(content, msg_type, tokenizer_ids)
        
        assert message.content == content
        assert message.type == msg_type
        assert message.tokenizer_ids == tokenizer_ids
    
    def test_message_creation_all_types(self):
        """Test message creation with all MessageType values."""
        content = "Test"
        
        for msg_type in MessageType:
            message = Message(content, msg_type)
            assert message.content == content
            assert message.type == msg_type


class TestRollout:
    """Test the Rollout class."""
    
    def test_rollout_initialization(self):
        """Test rollout initialization."""
        rollout = Rollout()
        assert rollout.messages == []
        assert isinstance(rollout.messages, list)
    
    def test_add_message_single(self):
        """Test adding a single message to rollout."""
        rollout = Rollout()
        message = Message("Hello", MessageType.SYSTEM)
        
        rollout.add_message(message)
        
        assert len(rollout.messages) == 1
        assert rollout.messages[0] == message
    
    def test_add_message_multiple(self):
        """Test adding multiple messages to rollout."""
        rollout = Rollout()
        message1 = Message("First", MessageType.SYSTEM)
        message2 = Message("Second", MessageType.PROMPT)
        message3 = Message("Third", MessageType.MODEL)
        
        rollout.add_message(message1)
        rollout.add_message(message2)
        rollout.add_message(message3)
        
        assert len(rollout.messages) == 3
        assert rollout.messages[0] == message1
        assert rollout.messages[1] == message2
        assert rollout.messages[2] == message3
    
    def test_format_conversation_empty(self):
        """Test format_conversation with empty rollout."""
        rollout = Rollout()
        formatted = rollout.format_conversation()
        
        assert formatted == []
        assert isinstance(formatted, list)
    
    def test_format_conversation_system_message(self):
        """Test format_conversation with system message."""
        rollout = Rollout()
        message = Message("System prompt", MessageType.SYSTEM)
        rollout.add_message(message)
        
        formatted = rollout.format_conversation()
        
        assert len(formatted) == 1
        assert formatted[0] == {"role": "system", "content": "System prompt"}
    
    def test_format_conversation_prompt_message(self):
        """Test format_conversation with prompt message."""
        rollout = Rollout()
        message = Message("User prompt", MessageType.PROMPT)
        rollout.add_message(message)
        
        formatted = rollout.format_conversation()
        
        assert len(formatted) == 1
        assert formatted[0] == {"role": "user", "content": "User prompt"}
    
    def test_format_conversation_user_message(self):
        """Test format_conversation with MESSAGE type."""
        rollout = Rollout()
        message = Message("User message", MessageType.MESSAGE)
        rollout.add_message(message)
        
        formatted = rollout.format_conversation()
        
        assert len(formatted) == 1
        assert formatted[0] == {"role": "user", "content": "User message"}
    
    def test_format_conversation_model_message(self):
        """Test format_conversation with model message."""
        rollout = Rollout()
        message = Message("Model response", MessageType.MODEL)
        rollout.add_message(message)
        
        formatted = rollout.format_conversation()
        
        assert len(formatted) == 1
        assert formatted[0] == {"role": "assistant", "content": "Model response"}
    
    def test_format_conversation_multiple_messages(self):
        """Test format_conversation with multiple message types."""
        rollout = Rollout()
        messages = [
            Message("System setup", MessageType.SYSTEM),
            Message("User question", MessageType.PROMPT),
            Message("User follow-up", MessageType.MESSAGE),
            Message("Model answer", MessageType.MODEL)
        ]
        
        for msg in messages:
            rollout.add_message(msg)
        
        formatted = rollout.format_conversation()
        
        expected = [
            {"role": "system", "content": "System setup"},
            {"role": "user", "content": "User question"},
            {"role": "user", "content": "User follow-up"},
            {"role": "assistant", "content": "Model answer"}
        ]
        
        assert formatted == expected
    
    def test_format_conversation_str_bug(self):
        """Test that reveals the bug in format_conversation_str."""
        rollout = Rollout()
        message = Message("Test", MessageType.SYSTEM)
        rollout.add_message(message)
        
        # This should fail because formatted_messages is a dict, not an object
        with pytest.raises(AttributeError, match="'dict' object has no attribute 'role'"):
            rollout.format_conversation_str()


class TestBugIdentification:
    """Tests that specifically identify bugs in the current implementation."""
    
    def test_missing_output_message_type(self):
        """Test that reveals the missing OUTPUT MessageType."""
        # This will cause an error because MessageType.OUTPUT doesn't exist
        # but it's referenced in format_conversation
        with pytest.raises(AttributeError, match="has no attribute 'OUTPUT'"):
            _ = MessageType.OUTPUT


# Integration tests
class TestIntegration:
    """Integration tests for the message module."""
    
    def test_full_conversation_flow(self):
        """Test a complete conversation flow."""
        rollout = Rollout()
        
        # Add messages in sequence
        system_msg = Message("You are a helpful assistant", MessageType.SYSTEM)
        user_msg = Message("What is 2+2?", MessageType.PROMPT)
        assistant_msg = Message("2+2 equals 4", MessageType.MODEL)
        
        rollout.add_message(system_msg)
        rollout.add_message(user_msg)
        rollout.add_message(assistant_msg)
        
        # Test that formatting works
        formatted = rollout.format_conversation()
        assert len(formatted) == 3
        assert all(isinstance(msg, dict) for msg in formatted)
        assert all("role" in msg and "content" in msg for msg in formatted)
        
        # Verify the conversation structure
        roles = [msg["role"] for msg in formatted]
        contents = [msg["content"] for msg in formatted]
        
        assert roles == ["system", "user", "assistant"]
        assert contents == [
            "You are a helpful assistant",
            "What is 2+2?", 
            "2+2 equals 4"
        ]



from typing import List

from dataclasses import dataclass
from enum import Enum


class MessageType(Enum):
    SYSTEM = "system"
    PROMPT = "prompt"
    MESSAGE = "user"
    MODEL = "assistant"



@dataclass
class Message:
    
    content: str
    type: MessageType
    tokenizer_ids: List[int] = None




    
class Rollout:
    def __init__(self):
        self.messages: List[Message] = []
        

    def add_message(self, message: Message, ):
        """Add a message to the conversation."""
        self.messages.append(message)

    def format_conversation(self) -> List[dict]:
        """
        Format conversation for Llama chat completion API.
        
        Returns a list of message dictionaries with 'role' and 'content' keys.
        """
        formatted_messages = []
        
        for message in self.messages:
            # Map MessageType to chat roles
            if message.type == MessageType.SYSTEM:
                role = "system"
            elif message.type == MessageType.PROMPT:
                role = "user"
            elif message.type == MessageType.MESSAGE:
                role = "user"
            elif message.type == MessageType.MODEL:
                role = "assistant"
            else:
                role = "user"  # default fallback
            
            formatted_messages.append({
                "role": role,
                "content": message.content
            })
        
        return formatted_messages

    def format_conversation_str(self) -> str:

        output = ""

        for formatted_messages in self.format_conversation():
            output += f"{formatted_messages['role']}: {formatted_messages['content']}"
        
        return output
    



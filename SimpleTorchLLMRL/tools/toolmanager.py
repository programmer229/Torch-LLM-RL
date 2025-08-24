

from typing import List


from .tool import Tool
from SimpleTorchLLMRL.chat.message import Message, MessageType

class ToolManger:


    def __init__(self, tools: List[Tool]) -> None:
        self.tools = tools

    
    def _parse_message(self, Rollout):
        
        pass

    def tool_explanation_prompt(self) -> str:
        
        tools= "\n".join(tool.explanation for tool in self.tools)
        general = f"""You are given the following tools to use: {tools}"""
        return general

    def process(self, message:Message) -> Message:

        if message.type != MessageType.MODEL:
            raise ValueError("Tool call should me done on user message")
        
        response_message = []

        for tool in self.tools:
            response = tool(message)
            if response: response_message.append(response)
        
        output = "\n".join(response_message)
        return Message(content=output, type=MessageType.SYSTEM)

            

        
        



        

    
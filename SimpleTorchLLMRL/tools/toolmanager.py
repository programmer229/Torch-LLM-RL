

from typing import List


from .tool import Tool
from SimpleTorchLLMRL.chat.message import Message, MessageType

class ToolManger:


    def __init__(self, tools: List[Tool]) -> None:
        self.tools = tools

    
    def _parse_message(self, Rollout):
    
        pass

    
    def tool_explanation_prompt(self):
        
        descriptions = '\n'.join(tool.get_description() for tool in self.tools)

        base_prompt = f"""
        You have the following tool available for you to use. You can call them by outputting there 
        tags then inside the tag with the content you want to pass
        {descriptions}
        """
        

        

        



    
    def process(self, message:Message):

        if message.type != MessageType.MODEL:
            raise ValueError("Tool call should me done on user message")

        



        

    
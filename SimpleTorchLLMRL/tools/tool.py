
from abc import ABC, abstractmethod
from typing import Any, Optional

from SimpleTorchLLMRL.chat.message import Message
from SimpleTorchLLMRL.utils.parse.tags import tag_parse

class Tool(ABC):


    def __init__(self, tags, description, name) -> None:
        """
        Initialize a Tool instance.
        
        ARGS:
            tags (str or list): Tags used for identifying and categorizing the tool
            instructions (str): Base instructions for how to use the tool
            
        """
        self._tags = tags
        self._description = description
        self._name = name
        
        tag_explanation = f"\nTo use the tool use the following output <{tags}> input  to function</{tags}>"
        self._instructions = description + tag_explanation

    @property
    def explanation(self):
        return self._instructions

    @abstractmethod
    def _execute(self, input) -> str | None: pass 

    def format_str_output(self, outputs: list[str]) -> str:
        output_str = f"Outputs form the {self._name}"
        for index, output in enumerate(outputs):
            output_str += f"Ouput {index+1}: {output}\n"
        return output_str

    
    def __call__(self, message:Message) -> str | None:
        input_string = message.content

        tool_calls = tag_parse(inside_tags=self._tags, text=input_string)
        
        
        outputs = []
        for tool_call in tool_calls:
            output = self._execute(tool_call)
            if output:
                outputs.append(output)
        

        return self.format_str_output(outputs) if outputs else None



from abc import ABC, abstractmethod
from typing import Any



class Tool(ABC):


    def __init__(self, tags, instructions) -> None:
        """
        Initialize a Tool instance.
        
        ARGS:
            tags (str or list): Tags used for identifying and categorizing the tool
            instructions (str): Base instructions for how to use the tool
            
        """
        self.tags = tags

        tag_explanation = f"\nTo use the tool use the following output <{tag}> input  to function</{tag}>"
        self.instructions = instructions + tag_explanation

    
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
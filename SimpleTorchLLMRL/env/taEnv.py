

from abc import ABC, abstractmethod
import random
from copy import deepcopy
from typing import Optional

try:
    import nltk  # type: ignore
except ImportError:
    print("nltk is not installed. Please install it with `uv pip install nltk`.")
    exit(1)

# monkey-patch nltk.download to always be quiet before importing textarena
_original_nltk_download = nltk.download
nltk.download = lambda *args, **kwargs: _original_nltk_download(
    *args, **{**kwargs, "quiet": True}
)

try:
    import textarena as ta  # type: ignore
except ImportError:
    print("textarena is not installed. Please install it with `uv pip install textarena`.")
    exit(1)

from SimpleTorchLLMRL.chat.message import Message, MessageType, Rollout
from SimpleTorchLLMRL.utils.typing import State
from .env import Env




class TAEnv(Env, ABC):
    """
    Wrapper for TextArena environments using SimpleTorchLLMRL interface.
    """

    def __init__(
        self,
        game: str = "Wordle-v0",
        seed: int = 0,
        system_prompt: Optional[str] = None
    ) -> None:
        super().__init__()
        self.game = game
        self.seed = seed
        self.system_prompt = system_prompt
        self.ta_env = ta.make(env_id=game)
        
        


    @abstractmethod
    def _game_setup(self, state: State) -> State:   
        """
        Game-specific setup logic that subclasses must implement.
        
        Args:
            state: The initial state dict
            
        Returns:
            State: Modified state with game-specific configuration
        """
        pass
    

    def setup(self) -> tuple[list[Message], State]:
        """Initialize the environment and return initial messages and state."""
        # Reset the environment
        self.ta_env.reset(num_players=1)
        
        # Get initial observation
        _, initial_observation = self.ta_env.get_observation()
        
        # Create initial messages
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append(Message(
                content=self.system_prompt,
                type=MessageType.SYSTEM
            ))
        
        # Add the initial game prompt
        messages.append(Message(
            content=initial_observation,
            type=MessageType.MESSAGE
        ))
        
        # Initialize state
        state: State = {
            "ta_env": deepcopy(self.ta_env),
            "is_finished": False,
            "seed": self.seed
        }
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        
        # Call game-specific setup (implemented by subclasses)
        state = self._game_setup(state)
        
        return messages, state

    @abstractmethod
    def _parse_user_action(self, rollout: Rollout) -> str:
        pass

    def get_env_response(self, rollout: Rollout, state: State) -> tuple[Message, State]:
        """Handle user action and return environment response."""
        # Get the TextArena environment from state
        ta_env = state["ta_env"]
        
        # Parse the user's action/guess
        user_action = self._parse_user_action(rollout)
        
        # Step the environment with the user's action
        is_finished, _ = ta_env.step(str(user_action))
        
        # Update state
        state["is_finished"] = is_finished
        
        # Get the environment's observation/feedback
        _, observation = ta_env.get_observation()
        
        # Create response message
        response_message = Message(
            content=observation,
            type=MessageType.MESSAGE
        )
        
        return response_message, state

    def is_complete(self, state: State) -> bool:
        """Check if the game/environment is finished."""
        return state.get("is_finished", False)



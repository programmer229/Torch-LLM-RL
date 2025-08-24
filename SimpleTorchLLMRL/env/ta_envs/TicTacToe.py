


from SimpleTorchLLMRL.utils.typing import State
from ..taEnv import TAEnv



class TicTacToe(TAEnv):



    def __init__(self, game: str = "Wordle-v0", seed: int = 0, system_prompt: str | None = None) -> None:

        super().__init__(game, seed, system_prompt)

    
    def _game_setup(self, state:State):
        pass


    
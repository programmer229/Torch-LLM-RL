


from SimpleTorchLLMRL.env.ToolQASolve import ToolUseEnv
from SimpleTorchLLMRL.chat.message import Rollout, Message, MessageType

from SimpleTorchLLMRL.tools import Calculator


def test_calculator():

    env = ToolUseEnv(
        custom_sys_prompt="Solve the following Math problems:",
        tools = [Calculator()]
    )
    rollout = Rollout()
    sys_message, state  = env.setup("","")
    rollout.add_messages(*sys_message)
    message = Message(content="<calculator>2+2</calculator>", type=MessageType.MODEL)
    rollout.add_messages(message)
    response = env.get_response_to_model(rollout, state)
    assert response




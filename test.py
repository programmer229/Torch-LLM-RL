        
# TRL(model= ..., trainer =grpo, rewardfunc= , tools= ..., mutliagent = ...)




#Let's implement GRPO in parts then abstract aways

from pickletools import optimize
from datasets import load_dataset
from torch.nn import parameter
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from copy import deepcopy
from tqdm import tqdm
import random

from SimpleTorchLLMRL.model.generate import ModelGenerate
from SimpleTorchLLMRL.env.QASolve import QASolverEnv
from SimpleTorchLLMRL.env.ToolQASolve import ToolUseEnv
from SimpleTorchLLMRL.chat.message import Rollout, Message, MessageType
from SimpleTorchLLMRL.trainer.GRPO import GRPO
from SimpleTorchLLMRL.reward.length import length_penalty
from SimpleTorchLLMRL.dataset.dataset import Dataset
from SimpleTorchLLMRL.tools import Calculator

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch




env = ToolUseEnv(
    custom_sys_prompt="Solve the following Math problems:",
    tools = [Calculator()]
)




rollout = Rollout()
sys_message, state  = env.setup("","")
rollout.add_messages(*sys_message)
message = Message(content="<calculator>2+2</calculator>", type=MessageType.MODEL)
rollout.add_messages(message)
response_message, state = env.get_response_to_model(rollout, state)
print(response_message)








# model_generator = ModelGenerate(
#     model=model, 
#     tokenizer=tokenizer,
#     max_new_tokens=100,
#     temperature=0.7
# )

# # Training hyperparameters
# batch_size = 8
# num_epochs = 3
# save_every = 100  # Save model every N steps

# # Training loop
# step = 0
# total_loss = 0
# running_loss = 0

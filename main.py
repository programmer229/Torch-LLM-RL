        
# TRL(model= ..., trainer =grpo, rewardfunc= , tools= ..., mutliagent = ...)




#Let's implement GRPO in parts then abstract aways

from pickletools import optimize
from datasets import load_dataset
from torch.nn import parameter
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from copy import deepcopy

from AgentOrchestration.model.generate import ModelGenerate
from AgentOrchestration.env.QASolve import QASolverEnv
from AgentOrchestration.chat.message import Rollout, Message, MessageType
from AgentOrchestration.trainer.GRPO import GRPO
from AgentOrchestration.reward.boxed import BoxedReward


# Load pre-trained model and tokenizer
model_name = "gpt2"  # or "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token for GPT2
model = GPT2LMHeadModel.from_pretrained(model_name)

ref_model = GPT2LMHeadModel.from_pretrained(model_name)


#Dataset loader
dataset = load_dataset("trl-lib/tldr", split="train")


#Train
trainer = GRPO(model = model, tokenizer= tokenizer, eps = 0.01)
optimizer = torch.optim.Adam(model.parameters())

reward = BoxedReward()

env = QASolverEnv(
    reward=reward,
    dataset = dataset,
    custom_sys_prompt = "Solve the following Math problems:"

)

model_generate = ModelGenerate(
                    model = model, 
                    tokenizer= tokenizer)


batch_size = 8

rollout = Rollout()

quesiton = "2+2"
ground_truth  = "4"

initial_message, state = env.setup(quesiton, ground_truth)

rollout.add_messages(*initial_message)

model_response = model_generate.rollout_generate_response(rollout)
print(model_response)

rollout.add_messages(model_response)

rollouts = [rollout]

rewards = reward(rollouts, [ground_truth])

loss = trainer.calculate_loss(rollouts=rollouts, rewards=rewards)

optimizer.zero_grad()
loss.backward() 

optimizer.step()

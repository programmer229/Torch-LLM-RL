        
# TRL(model= ..., trainer =grpo, rewardfunc= , tools= ..., mutliagent = ...)




#Let's implement GRPO in parts then abstract aways

from pickletools import optimize
from datasets import load_dataset
from torch.nn import parameter
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from copy import deepcopy

from SimpleTorchLLMRL.model.generate import ModelGenerate
from SimpleTorchLLMRL.env.QASolve import QASolverEnv
from SimpleTorchLLMRL.chat.message import Rollout, Message, MessageType
from SimpleTorchLLMRL.trainer.GRPO import GRPO
from SimpleTorchLLMRL.reward.boxed import BoxedReward
from SimpleTorchLLMRL.dataset.dataset import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"  # or "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token for GPT2
model = GPT2LMHeadModel.from_pretrained(model_name)


#Dataset loader
dataset = Dataset.from_huggingface(
            dataset_name="trl-lib/tldr",
            question_col="prompt",
            solution_col= "completion",
            split = "train"
            )


#Train
trainer = GRPO(model = model, tokenizer= tokenizer, eps = 0.01)
optimizer = torch.optim.Adam(model.parameters())

reward = BoxedReward()

env = QASolverEnv(
    reward=reward,
    dataset = dataset,
    custom_sys_prompt = "Solve the following Math problems:"
)

model_generater = ModelGenerate(
                    model = model, 
                    tokenizer= tokenizer)

batch_size = 8


rollout = Rollout()

quesiton = "2+2"
ground_truth  = "4"

initial_message, state = env.setup(quesiton, ground_truth)
rollout.add_messages(*initial_message)

model_response = model_generater.rollout_generate_response(rollout)
rollout.add_messages(model_response)

rollouts = [rollout]
rewards = reward(rollouts, [ground_truth])

loss = trainer.calculate_loss(rollouts=rollouts, rewards=rewards)

optimizer.zero_grad()
loss.backward() 
optimizer.step()

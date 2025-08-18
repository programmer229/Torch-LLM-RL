        
# TRL(model= ..., trainer =grpo, rewardfunc= , tools= ..., mutliagent = ...)




#Let's implement GRPO in parts then abstract aways

from pickletools import optimize
from datasets import load_dataset
from torch.nn import parameter
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

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



#Message Rollouts 
rollout = Rollout()
message = Message("calculate 2+ 32x =34", type=MessageType.PROMPT)
rollout.add_message(message)
message = Message("Ok I'm going to call calculator <calculator>...</calculator>", type=MessageType.MODEL)
rollout.add_message(message)
message = Message("Calculator output x=1", type=MessageType.SYSTEM)
rollout.add_message(message)
message = Message("Ohh ok the answer is boxed{x=1}", type=MessageType.MODEL)
                    
rollout.add_message(message)


rollouts = [rollout]*8  # List of Rollout objects, not tensor

groundth_truth = torch.ones(8)

rewards = BoxedReward()(rollouts, groundth_truth)

print(f"Number of rollouts: {len(rollouts)}")
print(f"First rollout messages: {len(rollouts[0].messages)}")

#Train
trainer = GRPO(model = model, tokenizer= tokenizer, eps = 0.01)
# reward = 
optimizer= torch.optim.Adam(model.parameters())
loss = trainer.calculate_loss(rollouts=rollouts, rewards=rewards)

optimizer.zero_grad()
loss.backward()  
optimizer.step()








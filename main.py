        
# TRL(model= ..., trainer =grpo, rewardfunc= , tools= ..., mutliagent = ...)




#Let's implement GRPO in parts then abstract aways

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from AgentOrchestration.utils.message import Rollout, Message, MessageType
from AgentOrchestration.trainer.GRPO import GRPO

# Model Set
model = AutoModel.from_pretrained("GPT-2")
tokenizer = AutoTokenizer.from_pretrained("GPT-2")


#Dataset loader
dataset = load_dataset("trl-lib/tldr", split="train")



#Message Rollouts 
message = Message("calculate 2+ 32x =34", type=MessageType.PROMPT)
Rollout.add_message(message)
message = Message("Ok I'm going to call calculator <calculator>...</calculator>", type=MessageType.MODEL)
Rollout.add_message(message)
message = Message("Calculator output x=1", type=MessageType.SYSTEM)
Rollout.add_message(message)
message = Message("Ohh ok the answer is boxed{x=1}", type=MessageType.MODEL)
                    
Rollout.add_message(message)

Rollouts = [Rollout]*8

#Train
GRPO(model, tokenizer)

reward = 
loss = GRPO.calculate_loss()





















 








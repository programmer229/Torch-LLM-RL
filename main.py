        
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
from SimpleTorchLLMRL.chat.message import Rollout, Message, MessageType
from SimpleTorchLLMRL.trainer.GRPO import GRPO
from SimpleTorchLLMRL.reward.length import length_penalty
from SimpleTorchLLMRL.dataset.dataset import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model setup
model_name = "gpt2"  # or "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token for GPT2
model = GPT2LMHeadModel.from_pretrained(model_name)

# Dataset loader
dataset = Dataset.from_huggingface(
            dataset_name="trl-lib/tldr",
            question_col="prompt",
            solution_col= "completion",
            split = "train"
            )

# Training components
trainer = GRPO(model=model, tokenizer=tokenizer, eps=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
reward = length_penalty

env = QASolverEnv(
    custom_sys_prompt="Solve the following Math problems:"
)

model_generator = ModelGenerate(
    model=model, 
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7
)

# Training hyperparameters
batch_size = 8
num_epochs = 3
save_every = 100  # Save model every N steps

# Training loop
step = 0
total_loss = 0
running_loss = 0

print(f"Starting training for {num_epochs} epochs...")
print(f"Dataset size: {len(dataset)}")
print(f"Batch size: {batch_size}")

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
    
    # Shuffle dataset indices for each epoch
    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)
    
    # Create batches
    num_batches = len(dataset_indices) // batch_size
    epoch_loss = 0
    
    with tqdm(range(num_batches), desc=f"Epoch {epoch + 1}") as pbar:
        for batch_idx in pbar:
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset_indices))
            batch_indices = dataset_indices[start_idx:end_idx]
            
            # Generate rollouts for this batch
            rollouts = []
            ground_truths = []
            
            for idx in batch_indices:
                data_item = dataset[idx]
                question = data_item["question"]
                ground_truth = data_item["solution"]
                
                # Create rollout
                rollout = Rollout()
                initial_messages, state = env.setup(question, ground_truth)
                rollout.add_messages(*initial_messages)
                
                # Generate model response
                model_response = model_generator.rollout_generate_response(rollout)
                rollout.add_messages(model_response)
                
                rollouts.append(rollout)
                ground_truths.append(ground_truth)
            
            # Calculate rewards
            rewards = reward(rollouts, ground_truths)
            print(rewards)
            
            # Calculate loss
            loss = trainer.calculate_loss(rollouts=rollouts, rewards=rewards)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            step += 1
            batch_loss = loss.item()
            total_loss += batch_loss
            running_loss += batch_loss
            epoch_loss += batch_loss
            print(loss.item())
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{running_loss / step:.4f}',
                'rewards': f'{rewards.mean().item():.3f}'
            })
            
print(f"\nTraining completed!")
print(f"Total steps: {step}")
print(f"Final average loss: {total_loss / step:.4f}")
# Training completed
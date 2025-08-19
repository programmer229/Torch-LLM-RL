


from pickletools import optimize
from datasets import load_dataset
from torch.nn import parameter
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pytest

from AgentOrchestration.chat.message import Rollout, Message, MessageType
from AgentOrchestration.trainer.GRPO import GRPO
from AgentOrchestration.reward.boxed import BoxedReward



@pytest.mark.slow
def test_rollout_grpo_backprop():
    """Integration test for the full training pipeline - marked as slow."""
    
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Message Rollouts 
    rollout = Rollout()
    message = Message("calculate 2+ 32x =34", type=MessageType.MESSAGE)
    rollout.add_messages(message)
    message = Message("Ok I'm going to call calculator <calculator>...</calculator>", type=MessageType.MODEL)
    rollout.add_messages(message)
    message = Message("Calculator output x=1", type=MessageType.SYSTEM)
    rollout.add_messages(message)
    message = Message("Ohh ok the answer is boxed{x=1}", type=MessageType.MODEL)
    rollout.add_messages(message)

    rollouts = [rollout]*8
    groundth_truth = torch.ones(8)
    rewards = BoxedReward()(rollouts, groundth_truth)

    # Train
    trainer = GRPO(model=model, tokenizer=tokenizer, eps=0.01)
    optimizer = torch.optim.Adam(model.parameters())
    loss = trainer.calculate_loss(rollouts=rollouts, rewards=rewards)

    optimizer.zero_grad()
    loss.backward()  
    optimizer.step()
    
    # Better assertions
    assert loss is not None







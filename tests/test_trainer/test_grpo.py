

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pytest

from SimpleTorchLLMRL.trainer.GRPO import GRPO

# Load pre-trained model and tokenizer







# ADVANTAGE
def test_grpo_advantage_all_ones():
    grpo = GRPO(None,None,None)
    n = 10
    rewards = torch.ones(n)
    advantage = grpo._calculate_advantage(rewards)
    assert torch.allclose(advantage,torch.zeros(n))

def test_grpo_advantage_all_zeros():
    grpo = GRPO(None,None,None)
    n = 10
    rewards = torch.zeros(n)
    advantage = grpo._calculate_advantage(rewards)
    assert torch.allclose(advantage,torch.zeros(n))

def test_grpo_one_rollout():
    grpo = GRPO(None,None,None)
    n = 1
    rewards = torch.ones(n)
    with pytest.raises(ValueError):
        grpo._calculate_advantage(rewards)


# RATIOS

def test_prob_ratios_same_model():
    model_name = "gpt2"  
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    ref_model = GPT2LMHeadModel.from_pretrained(model_name)

    grpo = GRPO(model, ref_model, 0)
    input_ids = torch.tensor(tokenizer.encode("Hello World!")).unsqueeze(0)
    probs_ratio = grpo._calculate_prob_ratios(input_ids)
    expected_output = torch.ones([1,len(input_ids), tokenizer.vocab_size]).float()
    assert torch.allclose(probs_ratio, expected_output)


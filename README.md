# SimpleTorchLLMRL

A PyTorch-native reinforcement learning library for training language models that embraces PyTorch's design philosophy while providing powerful abstractions for complex RL scenarios.

## 🎯 Philosophy

Unlike other RL libraries that abstract away PyTorch, SimpleTorchLLMRL is built **with** PyTorch, not **on top of** it. We provide the building blocks you need while letting you maintain full control over your training loop, model architecture, and optimization strategy.

## 🚀 Key Features

- **PyTorch-Native**: Direct integration with PyTorch training loops
- **Flexible Rollout System**: Conversation-based data structures that support complex interactions
- **Tool Integration**: Built-in support for function calling and external tools
- **Multi-Agent Support**: Natural extension to multi-agent scenarios
- **Modular Design**: Mix and match components as needed
- **Router-R1 Training**: Drop-in PPO trainer, reward shaping, and logging utilities to reproduce the Router-R1 strategy on top of SimpleTorchLLMRL components



## 🏗️ Core Architecture

### The Rollout System

At the heart of SimpleTorchLLMRL is the **Rollout** - a conversation-like data structure that captures the full interaction between agents, tools, and environments.

```python
from SimpleTorchLLMRL.chat.message import Rollout, Message, MessageType

# Create a rollout to track an interaction
rollout = Rollout()

# System message sets the context
rollout.add_messages(Message("You are a helpful math tutor", MessageType.SYSTEM))

# User provides input
rollout.add_messages(Message("What is 2 + 3?", MessageType.MESSAGE))

# Model generates response
rollout.add_messages(Message("2 + 3 = 5", MessageType.MODEL))
```

### Why Rollouts?

Rollouts provide several key advantages:

1. **Conversation Continuity**: Maintains full context across multi-turn interactions
2. **Tool Integration**: Seamlessly handles tool calls and responses within the conversation
3. **Multi-Agent Support**: Natural extension to conversations between multiple agents
4. **Flexible Evaluation**: Rich context for reward calculation and policy updates
5. **Selective Training**: Only messages marked as `MessageType.MODEL` are backpropagated during training, giving you precise control over what the model learns from

## 🛠️ Components

### 1. Message System
```python
# Three core message types
MessageType.SYSTEM    # System prompts, instructions - not trained on
MessageType.MESSAGE   # User/human input - not trained on  
MessageType.MODEL     # AI model responses - ONLY these are backpropagated
```

The key insight: **only `MessageType.MODEL` messages contribute to the training loss**. This allows you to include rich context (system prompts, user inputs, tool responses) without training the model to generate them.

### 2. Model Generation
```python
from SimpleTorchLLMRL.model.generate import ModelGenerate

generator = ModelGenerate(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7
)

# Single generation
response = generator.rollout_generate_response(rollout)

# Batch generation for efficiency
generator.batch_rollout_generate_response(rollouts)
```

### 3. Training with GRPO
```python
from SimpleTorchLLMRL.trainer.GRPO import GRPO

trainer = GRPO(
    model=model,
    tokenizer=tokenizer,
    eps=0.2  # PPO clipping parameter
)

# Calculate policy gradients
loss = trainer.calculate_loss(rollouts=rollouts, rewards=rewards)
loss.backward()
optimizer.step()
```

### 4. Reward Functions
```python
from SimpleTorchLLMRL.reward.boxed import BoxedReward

reward_fn = BoxedReward()
rewards = reward_fn(rollouts, ground_truths)
```

### 5. Environments
```python
from SimpleTorchLLMRL.env.QASolve import QASolverEnv

env = QASolverEnv(custom_sys_prompt="Solve this math problem:")
initial_messages, state = env.setup(question, answer)
```

## 🎮 Quick Start Example

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from SimpleTorchLLMRL.chat.message import Rollout, Message, MessageType
from SimpleTorchLLMRL.model.generate import ModelGenerate
from SimpleTorchLLMRL.trainer.GRPO import GRPO
from SimpleTorchLLMRL.reward.boxed import BoxedReward

# Setup model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Initialize components
generator = ModelGenerate(model=model, tokenizer=tokenizer)
trainer = GRPO(model=model, tokenizer=tokenizer, eps=0.2)
reward_fn = BoxedReward()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
rollouts = []
ground_truths = []

# Generate rollouts
for question, answer in dataset:
    rollout = Rollout()
    rollout.add_messages(Message("Solve this problem:", MessageType.SYSTEM))
    rollout.add_messages(Message(question, MessageType.MESSAGE))
    
    # Generate model response
    response = generator.rollout_generate_response(rollout)
    rollout.add_messages(response)
    
    rollouts.append(rollout)
    ground_truths.append(answer)

# Calculate rewards and train
rewards = reward_fn(rollouts, ground_truths)
loss = trainer.calculate_loss(rollouts, rewards)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 🔧 Tool Integration

The rollout system naturally supports tool usage with precise training control:

```python
# Tool call message - MODEL type, so this gets trained on
rollout.add_messages(Message("calculate(2 + 3)", MessageType.MODEL))

# Tool response - SYSTEM type, provides context but isn't trained on
rollout.add_messages(Message("Result: 5", MessageType.SYSTEM))

# Model continues based on tool result - MODEL type, gets trained on
rollout.add_messages(Message("The answer is 5", MessageType.MODEL))
```

**Training Behavior**: The model learns to make tool calls and use their results, but doesn't learn to generate the tool responses themselves - exactly what you want!

## 👥 Multi-Agent Scenarios

Extend conversations to multiple agents with selective training:

```python
# Agent A responds - MODEL type, gets trained on
rollout.add_messages(Message("I think the answer is 4", MessageType.MODEL))

# Agent B responds - MODEL type, gets trained on
rollout.add_messages(Message("Actually, let me double-check that calculation...", MessageType.MODEL))

# Tool usage by Agent B - MODEL type, learns to call tools
rollout.add_messages(Message("calculate(2 + 3)", MessageType.MODEL))

# Tool response - SYSTEM type, not trained on
rollout.add_messages(Message("Result: 5", MessageType.SYSTEM))

# Agent B final response - MODEL type, learns to use tool results
rollout.add_messages(Message("The correct answer is 5", MessageType.MODEL))
```

**Multi-Agent Training**: Each agent's `MessageType.MODEL` responses are included in training, allowing you to train multiple agents simultaneously or selectively train specific agents by controlling message types.

## 🏃‍♂️ Running the Example

```bash
# Run the full training example
python main.py

# Or with uv
uv run python main.py
```

The example demonstrates:
- Dataset loading and preprocessing
- Batch inference for efficiency
- GRPO training with reward optimization
- Model checkpointing and evaluation

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run core component tests only
uv run pytest tests/unit/test_chat/ tests/unit/test_reward/ tests/unit/test_trainer/ tests/unit/test_tools/ -v
```

## 📁 Project Structure

```
SimpleTorchLLMRL/
├── chat/           # Message and rollout system
├── env/            # Environment abstractions  
├── model/          # Model generation utilities
├── reward/         # Reward function implementations
├── tools/          # Tool integration system
├── trainer/        # Training algorithms (GRPO, etc.)
└── utils/          # Utility functions

tests/              # Comprehensive test suite
main.py            # Complete training example
```

## 🎯 Use Cases

- **Conversational AI Training**: Multi-turn dialogue optimization
- **Tool-Augmented Models**: Training models to effectively use external tools
- **Math Problem Solving**: Reward models based on correct solutions
- **Multi-Agent Collaboration**: Training agents to work together
- **Custom RL Scenarios**: Flexible foundation for novel RL applications

## 📄 License

MIT License - see LICENSE file for details


---
## 🛣️ Router-R1 Integration

The `SimpleTorchLLMRL.router_r1` package bundles the components required to reproduce the Router-R1 reinforcement learning recipe. It ships with:

- configuration dataclasses that mirror the original Hydra setup
- a generation manager that prepares `<think>`, `<search>`, `<information>`, and `<answer>` rollouts
- a rule-based reward function with exact-match and F1 scoring plus format penalties
- a PPO training loop that logs to console or Weights & Biases at configurable validation intervals

See `examples/router_r1_training.py` for a minimal toy walkthrough or `examples/router_r1_math_qwen.py` for a HendrycksMath + Qwen training harness similar to the original `train.sh` workflow.

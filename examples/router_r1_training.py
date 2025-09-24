"""Minimal example showing how to launch Router-R1 style training."""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

from SimpleTorchLLMRL import (
    RouterDataConfig,
    RouterGenerationConfig,
    RouterRewardConfig,
    RouterTrainerConfig,
    RouterTrainingConfig,
    RouterValidationConfig,
    RouterR1Trainer,
)
from SimpleTorchLLMRL.dataset.dataset import Dataset


SYSTEM_PROMPT = (
    "You are Router-R1, an AI assistant that must reason using <think>, "
    "optionally issue <search> instructions, synthesize <information> blocks, "
    "and conclude with a single <answer>."
)


def create_toy_dataset() -> Dataset:
    data = [
        {
            "question": "What is the capital of France?",
            "solution": "<think>Paris is the capital city of France.</think><answer>Paris</answer>",
        },
        {
            "question": "Solve: 2 + 2",
            "solution": "<think>Adding 2 and 2 gives 4.</think><answer>4</answer>",
        },
    ]
    return Dataset(data)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

    dataset = create_toy_dataset()

    training_config = RouterTrainingConfig(
        data=RouterDataConfig(system_prompt=SYSTEM_PROMPT, train_batch_size=2, val_batch_size=2),
        generation=RouterGenerationConfig(max_new_tokens=64, temperature=0.8, do_sample=True, top_p=0.95),
        reward=RouterRewardConfig(reward_metric="em", cost_coefficient=0.0),
        trainer=RouterTrainerConfig(total_epochs=1, total_training_steps=4, test_freq=2, logger_backends=["console"]),
        validation=RouterValidationConfig(max_batches=1, store_completions=True),
    )

    trainer = RouterR1Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        val_dataset=dataset,
        config=training_config,
    )
    trainer.fit()


if __name__ == "__main__":
    main()

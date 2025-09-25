"""Router-R1 style PPO training on HendrycksMath with a Qwen/DeepSeek model."""

from __future__ import annotations

import argparse
import ast
import re
from typing import List, Optional

import torch
from datasets import load_dataset
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
    "You are Router-R1, an expert math problem solver. You must use <think> to reason, "
    "optionally issue <search> queries, record retrieved facts with <information>, "
    "and provide exactly one final <answer>."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Router-R1 PPO on HendrycksMath using Qwen2.5 distill")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Actor/critic model name (HF hub path)")
    parser.add_argument("--reference-model", default=None,
                        help="Optional reference policy weights. Defaults to reloading --model on CPU.")
    parser.add_argument("--subject", default="algebra",
                        help="HendrycksMath subject split (e.g. algebra, geometry, number_theory)")
    parser.add_argument("--train-split", default="train", help="HF dataset split for training")
    parser.add_argument("--val-split", default="test", help="HF dataset split for validation")
    parser.add_argument("--train-samples", type=int, default=None,
                        help="Optional cap on training samples for quick experiments")
    parser.add_argument("--val-samples", type=int, default=512,
                        help="Optional cap on validation samples")
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--total-training-steps", type=int, default=225)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--test-freq", type=int, default=15)
    parser.add_argument("--val-before-train", action="store_true")
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--experiment-name", default="router-r1-hendrycks-qwen")
    parser.add_argument("--torch-dtype", default="bfloat16",
                        help="torch dtype for model loading (e.g. float16, bfloat16)")
    parser.add_argument("--device-map", default="auto",
                        help="device map passed to AutoModel.from_pretrained (e.g. auto, cuda, cpu)")
    parser.add_argument("--save-completions", action="store_true")
    parser.add_argument("--logger", default="['console']",
                        help="Tracker backends, e.g. ['console','wandb']")
    return parser.parse_args()


def extract_boxed_answer(solution: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", solution)
    if matches:
        return matches[-1].strip()
    cleaned = solution.strip().splitlines()
    return cleaned[-1].strip() if cleaned else solution.strip()


def build_dataset(subject: str, split: str, limit: Optional[int]) -> Dataset:
    hf_ds = load_dataset("hendrycks_math", subject, split=split)
    if limit is not None:
        hf_ds = hf_ds.select(range(min(limit, len(hf_ds))))

    data: List[dict] = []
    for item in hf_ds:
        question = item["problem"].strip()
        solution = item["solution"]
        answer = extract_boxed_answer(solution)
        data.append({"question": question, "solution": answer})
    return Dataset(data)


def _resolve_dtype(name: Optional[str]):
    if not name or name.lower() == "none":
        return None
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, name)


def load_model(name: str, torch_dtype: Optional[str], device_map: Optional[str]):
    model_kwargs = {"trust_remote_code": True}
    resolved_dtype = _resolve_dtype(torch_dtype)
    if resolved_dtype is not None:
        if device_map and device_map.lower() == "cpu" and resolved_dtype != torch.float32:
            model_kwargs["torch_dtype"] = torch.float32
        else:
            model_kwargs["torch_dtype"] = resolved_dtype
    if device_map and device_map.lower() != "none":
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    return model


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = load_model(args.model, args.torch_dtype, args.device_map)
    if getattr(model.config, "use_cache", True):
        model.config.use_cache = False

    reference_model = None
    if args.reference_model is not None:
        reference_model = load_model(args.reference_model, args.torch_dtype, "cpu")
    else:
        print("[INFO] Loading reference policy on CPU (may require large memory).")
        reference_model = load_model(args.model, args.torch_dtype, "cpu")

    train_dataset = build_dataset(args.subject, args.train_split, args.train_samples)
    val_dataset = build_dataset(args.subject, args.val_split, args.val_samples) if args.val_split else None

    trainer_config = RouterTrainingConfig(
        data=RouterDataConfig(
            system_prompt=SYSTEM_PROMPT,
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
        ),
        generation=RouterGenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
        reward=RouterRewardConfig(reward_metric="em", cost_coefficient=0.0),
        trainer=RouterTrainerConfig(
            total_epochs=args.total_epochs,
            total_training_steps=args.total_training_steps,
            val_before_train=args.val_before_train,
            test_freq=args.test_freq,
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            logger_backends=ast.literal_eval(args.logger),
        ),
        validation=RouterValidationConfig(store_completions=args.save_completions),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    trainer = RouterR1Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config,
        optimizer=optimizer,
        reference_model=reference_model,
    )
    trainer.fit()


if __name__ == "__main__":
    main()

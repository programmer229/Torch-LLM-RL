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
    RouterToolConfig,
    RouterTrainerConfig,
    RouterTrainingConfig,
    RouterValidationConfig,
    RouterR1Trainer,
)
from SimpleTorchLLMRL.dataset.dataset import Dataset

SYSTEM_PROMPT = (
    "Answer the given question.\n"
    "Every time you receive new information, you must first conduct reasoning inside <think> ... </think>.\n"
    "After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a query inside <search> LLM-Name:Your-Query </search>.\n"
    "\n"
    "!!! STRICT FORMAT RULES for <search>: !!!\n"
    "    + You MUST replace LLM-Name with the EXACT name of a model selected from [qwen2.5].\n"
    "    + You MUST replace Your-Query with a CONCRETE QUESTION that helps answer the original question below.\n"
    "    + NEVER copy or paste model descriptions into <search>.\n"
    "    + NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly.\n"
    "\n"
    "Before each LLM call, you MUST explicitly reason inside <think> ... </think> about:\n"
    "    + Why external information is needed.\n"
    "    + Which model is best suited for answering it, based on the LLMs' abilities (described below).\n"
    "\n"
    "When you call an LLM, the response will be returned between <information> and </information>.\n"
    "Only qwen2.5 is available in the routing pool, so you must never reference any other model name.\n"
    "Call it whenever additional context is needed, and do not fabricate alternative specialists.\n"
    "\n"
    "#### The Descriptions of Each LLM\n"
    "\n"
    "qwen2.5:\n"
    "qwen2.5 is a strong reasoning specialist focused on step-by-step analysis. Use it as the go-to tool for multi-step analysis and factual synthesis in this setup.\n"
    "\n"
    "If you find that no further external knowledge is needed, you can directly provide your final answer inside <answer> ... </answer>, without additional explanation or illustration.\n"
    "For example: <answer> Beijing </answer>.\n"
    "    + Important: You must not output the placeholder text \"<answer> and </answer>\" alone.\n"
    "    + You must insert your actual answer between <answer> and </answer>, following the correct format.\n"
    "Question: {question}\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Router-R1 PPO on HendrycksMath using Qwen2.5 distill")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Actor/critic model name (HF hub path)")
    parser.add_argument("--reference-model", default=None,
                        help="Optional reference policy weights. Defaults to reloading --model on CPU.")
    parser.add_argument("--dataset", default="EleutherAI/hendrycks_math",
                        help="HF dataset identifier for HendrycksMath")
    parser.add_argument("--subject", default="algebra",
                        help="HendrycksMath subject split (e.g. algebra, geometry, number_theory)")
    parser.add_argument("--train-split", default="train", help="HF dataset split for training")
    parser.add_argument("--val-split", default="test", help="HF dataset split for validation")
    parser.add_argument("--train-samples", type=int, default=None,
                        help="Optional cap on training samples for quick experiments")
    parser.add_argument("--val-samples", type=int, default=512,
                        help="Optional cap on validation samples")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
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
    parser.add_argument("--log-train-completions", action="store_true",
                        help="Print a sample of model completions each training step")
    parser.add_argument("--log-train-completions-file", default=None,
                        help="Optional path to append train completions for offline inspection")
    parser.add_argument("--tool-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model identifier for the routing tool (defaults to Qwen2.5)")
    parser.add_argument("--tool-device-map", default="auto",
                        help="Device map passed to AutoModel for the routing tool")
    parser.add_argument("--tool-dtype", default=None,
                        help="Optional torch dtype for the routing tool (e.g. float16, bfloat16)")
    return parser.parse_args()


def extract_boxed_answer(solution: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", solution)
    if matches:
        return matches[-1].strip()
    cleaned = solution.strip().splitlines()
    return cleaned[-1].strip() if cleaned else solution.strip()


def build_dataset(dataset_path: str, subject: str, split: str, limit: Optional[int]) -> Dataset:
    hf_ds = load_dataset(dataset_path, subject, split=split)
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

    train_dataset = build_dataset(args.dataset, args.subject, args.train_split, args.train_samples)
    val_dataset = (build_dataset(args.dataset, args.subject, args.val_split, args.val_samples)
                   if args.val_split else None)

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
        tool=RouterToolConfig(
            enabled=True,
            tool_name="qwen2.5",
            model_id=args.tool_model,
            device_map=args.tool_device_map,
            torch_dtype=args.tool_dtype,
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
            log_train_completions=args.log_train_completions,
            log_train_completions_file=args.log_train_completions_file,
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

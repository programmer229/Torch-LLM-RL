from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from SimpleTorchLLMRL.chat.message import Rollout
from SimpleTorchLLMRL.router_r1.config import RouterTrainingConfig
from SimpleTorchLLMRL.router_r1.generation import RouterGenerationManager
from SimpleTorchLLMRL.router_r1.reward import RouterReward, RouterRewardResult
from SimpleTorchLLMRL.router_r1.tracking import Tracking
from SimpleTorchLLMRL.trainer.PPO import PPO


@dataclass
class RouterBatch:
    questions: List[str]
    solutions: List[str]
    data_sources: Optional[List[str]] = None


class RouterR1Trainer:
    """High-level trainer that mirrors the Router-R1 PPO loop on top of SimpleTorchLLMRL."""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        config: RouterTrainingConfig,
        *,
        val_dataset: Optional[Dataset] = None,
        reward_fn: Optional[RouterReward] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        ppo: Optional[PPO] = None,
        reference_model=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = next(self.model.parameters()).device
        if not getattr(self.model, "hf_device_map", None):
            self.model.to(self.device)

        self.generation_manager = RouterGenerationManager(
            model=self.model,
            tokenizer=self.tokenizer,
            data_config=config.data,
            generation_config=config.generation,
        )
        self.reward_fn = reward_fn or RouterReward(
            reward_metric=config.reward.reward_metric,
            cost_coefficient=config.reward.cost_coefficient,
            format_penalty_small=config.reward.format_penalty_small,
            format_penalty_medium=config.reward.format_penalty_medium,
            format_penalty_large=config.reward.format_penalty_large,
        )
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.scheduler = scheduler
        self.ppo = ppo or PPO(model=self.model,
                              tokenizer=self.tokenizer,
                              reference_model=reference_model)
        self.ppo.model.train()

        self.tracking = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backends=config.trainer.logger_backends,
            config=asdict(config),
        )

        self.global_step = 0
        self._configure_dataloaders()

    def _configure_dataloaders(self) -> None:
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=self.config.data.shuffle,
            collate_fn=self._collate,
        )
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.data.val_batch_size,
                shuffle=False,
                collate_fn=self._collate,
            )
        else:
            self.val_dataloader = None

    @staticmethod
    def _collate(batch: Sequence[Dict[str, str]]) -> RouterBatch:
        questions = [item.get("question", "") for item in batch]
        solutions = [item.get("solution", "") for item in batch]
        sources = [item.get("data_source") for item in batch if "data_source" in item]
        data_sources = sources if sources else None
        return RouterBatch(questions=questions, solutions=solutions, data_sources=data_sources)

    def fit(self) -> None:
        if self.config.trainer.val_before_train and self.val_dataloader is not None:
            val_metrics = self.validate()
            self.tracking.log(val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        total_steps_target = self.config.trainer.total_training_steps
        for epoch in range(self.config.trainer.total_epochs):
            for batch in self.train_dataloader:
                self.model.train()
                metrics = self._training_step(batch)
                self.tracking.log(metrics, step=self.global_step)
                self.global_step += 1

                if self.config.trainer.test_freq > 0 and self.val_dataloader is not None:
                    if self.global_step % self.config.trainer.test_freq == 0:
                        val_metrics = self.validate()
                        self.tracking.log(val_metrics, step=self.global_step)

                if total_steps_target is not None and self.global_step >= total_steps_target:
                    if self.val_dataloader is not None:
                        val_metrics = self.validate()
                        self.tracking.log(val_metrics, step=self.global_step)
                    return
            if total_steps_target is None and self.val_dataloader is not None:
                val_metrics = self.validate()
                self.tracking.log(val_metrics, step=self.global_step)

    def _training_step(self, batch: RouterBatch) -> Dict[str, float]:
        rollouts = self.generation_manager.generate_from_questions(batch.questions)
        if self.config.trainer.log_train_completions:
            self._log_train_completions(rollouts)
        reward_result = self.reward_fn(rollouts, batch.solutions, state="train", data_sources=batch.data_sources)

        rewards = reward_result.rewards.to(self.device)
        loss = self.ppo.calculate_loss(rollouts, rewards)

        self.optimizer.zero_grad()
        loss.backward()

        if self.config.trainer.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.trainer.max_grad_norm)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        metrics = {
            "train/loss": float(loss.detach().cpu().item()),
            "train/reward": float(reward_result.rewards.mean().item()),
            "train/em": float(reward_result.em.mean().item()),
            "train/f1": float(reward_result.f1.mean().item()),
            "train/format_penalty": float(reward_result.format_penalty.mean().item()),
            "train/route_count": float(reward_result.route_counts.mean().item()),
        }

        return metrics

    def _log_train_completions(self, rollouts: Sequence[Rollout]) -> None:
        completions = self.generation_manager.extract_completions(rollouts)
        if not completions:
            print("[TRAIN] No completions produced this step.")
            return

        max_examples = 3
        for idx, completion in enumerate(completions[:max_examples]):
            print(f"[TRAIN] completion[{idx}]:\n{completion}\n---")

        file_path = self.config.trainer.log_train_completions_file
        if file_path:
            try:
                path_obj = Path(file_path)
                if not path_obj.parent.exists():
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                with path_obj.open("a", encoding="utf-8") as fh:
                    fh.write("=== TRAIN STEP " + str(self.global_step) + " ===\n")
                    for idx, completion in enumerate(completions):
                        fh.write(f"completion[{idx}]:\n{completion}\n\n")
            except OSError as exc:
                print(f"[WARN] Failed to write train completions to '{file_path}': {exc}")

    def validate(self) -> Dict[str, float]:
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        all_rewards: List[RouterRewardResult] = []
        completions: List[str] = []
        max_batches = self.config.validation.max_batches
        for batch_idx, batch in enumerate(self.val_dataloader):
            rollouts = self.generation_manager.generate_from_questions(batch.questions)
            reward_result = self.reward_fn(
                rollouts,
                batch.solutions,
                state="val",
                data_sources=batch.data_sources,
            )
            all_rewards.append(reward_result)
            if self.config.validation.store_completions:
                completions.extend(self.generation_manager.extract_completions(rollouts))
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

        self.model.train()

        if not all_rewards:
            return {}

        reward_tensor = torch.cat([result.rewards for result in all_rewards])
        em_tensor = torch.cat([result.em for result in all_rewards])
        f1_tensor = torch.cat([result.f1 for result in all_rewards])
        penalty_tensor = torch.cat([result.format_penalty for result in all_rewards])
        route_tensor = torch.cat([result.route_counts for result in all_rewards])

        metrics = {
            "val/reward": float(reward_tensor.mean().item()),
            "val/em": float(em_tensor.mean().item()),
            "val/f1": float(f1_tensor.mean().item()),
            "val/format_penalty": float(penalty_tensor.mean().item()),
            "val/route_count": float(route_tensor.mean().item()),
        }

        return metrics

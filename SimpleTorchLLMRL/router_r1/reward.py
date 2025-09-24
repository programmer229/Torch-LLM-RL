from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from SimpleTorchLLMRL.chat.message import MessageType, Rollout


_TAG_PATTERN = re.compile(r"<(search|answer|think|information)>(.*?)</\\1>", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class RouterRewardResult:
    rewards: torch.Tensor
    em: torch.Tensor
    f1: torch.Tensor
    format_penalty: torch.Tensor
    route_counts: torch.Tensor

    def to_dict(self, prefix: str) -> dict[str, float]:
        tensor_stats = {
            f"{prefix}/reward": float(self.rewards.mean().item()),
            f"{prefix}/em": float(self.em.mean().item()),
            f"{prefix}/f1": float(self.f1.mean().item()),
            f"{prefix}/format_penalty": float(self.format_penalty.mean().item()),
            f"{prefix}/route_count": float(self.route_counts.mean().item()),
        }
        return tensor_stats


class RouterReward:
    """Rule-based reward shaping used by Router-R1."""

    def __init__(
        self,
        *,
        reward_metric: str = "em",
        cost_coefficient: float = 0.0,
        format_penalty_small: float = -0.25,
        format_penalty_medium: float = -0.5,
        format_penalty_large: float = -1.0,
    ) -> None:
        allowed = {"em", "f1"}
        if reward_metric not in allowed:
            raise ValueError(f"reward_metric must be one of {allowed}")

        self.reward_metric = reward_metric
        self.cost_coefficient = cost_coefficient
        self.penalty_small = format_penalty_small
        self.penalty_medium = format_penalty_medium
        self.penalty_large = format_penalty_large

    def __call__(
        self,
        rollouts: Sequence[Rollout],
        ground_truth: Sequence[str],
        *,
        state: str = "train",
        data_sources: Optional[Sequence[str]] = None,
    ) -> RouterRewardResult:
        if len(rollouts) == 0:
            zeros = torch.zeros(0)
            return RouterRewardResult(zeros, zeros, zeros, zeros, zeros)

        device = None
        if hasattr(rollouts[0], "device"):
            device = rollouts[0].device  # type: ignore[attr-defined]

        em_scores: List[float] = []
        f1_scores: List[float] = []
        penalties: List[float] = []
        routes: List[float] = []
        rewards: List[float] = []

        for idx, rollout in enumerate(rollouts):
            completion = _extract_model_completion(rollout)
            answer_text = _extract_answer_text(completion)
            eval_text = answer_text if answer_text else completion
            truth = ground_truth[idx] if idx < len(ground_truth) else ""
            strict_penalty = self._format_reward(completion)
            em_score = _exact_match(eval_text, truth)
            f1_score = _f1(eval_text, truth)
            route_cnt = float(route_count(completion))

            metric_score = em_score if self.reward_metric == "em" else f1_score
            total_reward = metric_score + strict_penalty - self.cost_coefficient * route_cnt

            em_scores.append(em_score)
            f1_scores.append(f1_score)
            penalties.append(strict_penalty)
            routes.append(route_cnt)
            rewards.append(total_reward)

        tensor_kwargs = {"device": device} if device is not None else {}
        reward_tensor = torch.tensor(rewards, **tensor_kwargs)
        em_tensor = torch.tensor(em_scores, **tensor_kwargs)
        f1_tensor = torch.tensor(f1_scores, **tensor_kwargs)
        penalty_tensor = torch.tensor(penalties, **tensor_kwargs)
        route_tensor = torch.tensor(routes, **tensor_kwargs)

        return RouterRewardResult(
            rewards=reward_tensor,
            em=em_tensor,
            f1=f1_tensor,
            format_penalty=penalty_tensor,
            route_counts=route_tensor,
        )

    def _format_reward(self, completion: str) -> float:
        matches = list(_TAG_PATTERN.findall(completion))
        if not matches:
            return self.penalty_large

        tag_counts = {"search": 0, "answer": 0, "think": 0, "information": 0}
        query_format_punish = False
        llm_name_punish = False
        think_punish = False
        for tag, content in matches:
            content = content.strip()
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if tag == "search":
                if content.count(":") != 1:
                    query_format_punish = True
                else:
                    llm_name, query = [item.strip() for item in content.split(":", 1)]
                    if not llm_name:
                        llm_name_punish = True
                    if not query:
                        query_format_punish = True
            if tag == "think" and content in {"", "..."}:
                think_punish = True
            if _TAG_PATTERN.search(content):
                return self.penalty_large

        if think_punish:
            return self.penalty_large

        if tag_counts.get("answer", 0) != 1 or tag_counts.get("think", 0) == 0:
            return self.penalty_large

        if query_format_punish:
            return self.penalty_medium

        if llm_name_punish:
            return self.penalty_small

        return 0.0


def _extract_model_completion(rollout: Rollout) -> str:
    for message in reversed(list(rollout)):
        if message.type == MessageType.MODEL:
            return message.content
    return ""


def _extract_answer_text(completion: str) -> str:
    match = _ANSWER_PATTERN.search(completion)
    if match:
        return match.group(1).strip()
    return ""


def route_count(completion: str) -> int:
    matches = list(_TAG_PATTERN.findall(completion))
    if not matches:
        return 0

    valid_routes = 0
    for tag, content in matches:
        if tag != "search":
            continue
        if content.count(":") != 1:
            continue
        llm_name, query = [item.strip() for item in content.split(":", 1)]
        if not llm_name or not query:
            continue
        valid_routes += 1
    return valid_routes


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _exact_match(prediction: str, truth: str) -> float:
    return float(_normalize(prediction) == _normalize(truth)) if truth else 0.0


def _f1(prediction: str, truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    truth_tokens = _normalize(truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = 0
    truth_counts = {}
    for token in truth_tokens:
        truth_counts[token] = truth_counts.get(token, 0) + 1
    for token in pred_tokens:
        if truth_counts.get(token, 0) > 0:
            truth_counts[token] -= 1
            common += 1
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

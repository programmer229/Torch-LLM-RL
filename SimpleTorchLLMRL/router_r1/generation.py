from __future__ import annotations

from typing import Iterable, List, Sequence

from SimpleTorchLLMRL.chat.message import Message, MessageType, Rollout
from SimpleTorchLLMRL.model.generate import ModelGenerate
from SimpleTorchLLMRL.router_r1.config import RouterDataConfig, RouterGenerationConfig


class RouterGenerationManager:
    """Utility that prepares Router-style rollouts and triggers batched decoding."""

    def __init__(
        self,
        model,
        tokenizer,
        data_config: RouterDataConfig,
        generation_config: RouterGenerationConfig,
    ) -> None:
        self.data_config = data_config
        self.system_prompt = data_config.system_prompt
        self.generator = ModelGenerate(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            do_sample=generation_config.do_sample,
            top_p=generation_config.top_p,
        )
        self.tokenizer = tokenizer
        self.model = model

    def build_rollouts(self, questions: Sequence[str]) -> List[Rollout]:
        rollouts: List[Rollout] = []
        for question in questions:
            rollout = Rollout()
            if self.system_prompt:
                rollout.add_messages(Message(self.system_prompt, MessageType.SYSTEM))
            rollout.add_messages(Message(question, MessageType.MESSAGE))
            rollouts.append(rollout)
        return rollouts

    def generate(self, rollouts: Sequence[Rollout]) -> None:
        if not rollouts:
            return
        self.generator.batch_rollout_generate_response(list(rollouts))
        self.model.train()

    def generate_from_questions(self, questions: Sequence[str]) -> List[Rollout]:
        rollouts = self.build_rollouts(questions)
        self.generate(rollouts)
        return rollouts

    @staticmethod
    def extract_completions(rollouts: Sequence[Rollout]) -> List[str]:
        completions: List[str] = []
        for rollout in rollouts:
            completion = ""
            for message in reversed(list(rollout)):
                if message.type == MessageType.MODEL:
                    completion = message.content
                    break
            completions.append(completion)
        return completions

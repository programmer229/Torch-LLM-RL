from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from SimpleTorchLLMRL.chat.message import Message, MessageType, Rollout
from SimpleTorchLLMRL.model.generate import ModelGenerate
from SimpleTorchLLMRL.router_r1.config import (
    RouterDataConfig,
    RouterGenerationConfig,
    RouterToolConfig,
)
from SimpleTorchLLMRL.router_r1.tools import QwenToolInvoker


_SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class RouterGenerationManager:
    """Utility that prepares Router-style rollouts and runs multi-turn decoding."""

    def __init__(
        self,
        model,
        tokenizer,
        data_config: RouterDataConfig,
        generation_config: RouterGenerationConfig,
        *,
        tool_config: Optional[RouterToolConfig] = None,
    ) -> None:
        self.data_config = data_config
        self.system_prompt = data_config.system_prompt
        self.generation_config = generation_config
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
        self.tool_config = tool_config or RouterToolConfig()
        self.allowed_tools = {self.tool_config.tool_name.lower()} if self.tool_config.tool_name else set()
        self.tool_invoker: Optional[QwenToolInvoker] = None
        if self.tool_config.enabled:
            self.tool_invoker = QwenToolInvoker(self.tool_config)

    def build_rollouts(self, questions: Sequence[str]) -> List[Rollout]:
        rollouts: List[Rollout] = []
        for question in questions:
            rollout = Rollout()
            if self.system_prompt:
                rollout.add_messages(Message(self.system_prompt, MessageType.SYSTEM))
            rollout.add_messages(Message(question, MessageType.MESSAGE))
            rollouts.append(rollout)
        return rollouts

    def generate_from_questions(self, questions: Sequence[str]) -> List[Rollout]:
        rollouts = self.build_rollouts(questions)
        if not rollouts:
            return []

        active_indices = list(range(len(rollouts)))
        tool_usage = [0] * len(rollouts)

        for turn in range(self.generation_config.max_turns):
            if not active_indices:
                break

            active_rollouts = [rollouts[idx] for idx in active_indices]
            self.generator.batch_rollout_generate_response(active_rollouts)
            self.model.train()

            completions = self.extract_completions(active_rollouts)
            next_active: List[int] = []

            for local_idx, rollout_idx in enumerate(active_indices):
                completion = completions[local_idx]
                should_continue = self._process_completion(
                    rollout=rollouts[rollout_idx],
                    completion=completion,
                    rollout_idx=rollout_idx,
                    tool_usage=tool_usage,
                )
                if should_continue and (turn + 1) < self.generation_config.max_turns:
                    next_active.append(rollout_idx)

            active_indices = next_active

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

    def _process_completion(
        self,
        *,
        rollout: Rollout,
        completion: str,
        rollout_idx: int,
        tool_usage: List[int],
    ) -> bool:
        search_blocks = _SEARCH_PATTERN.findall(completion)
        for block in search_blocks:
            tool_name, query = self._parse_tool_invocation(block)
            info_response = self._invoke_tool(rollout_idx, tool_usage, tool_name, query)
            message = Message(f"<information>{info_response}</information>", MessageType.MESSAGE)
            rollout.add_messages(message)

        has_answer = bool(_ANSWER_PATTERN.search(completion))
        return not has_answer

    def _parse_tool_invocation(self, content: str) -> Tuple[Optional[str], str]:
        raw = content.strip()
        if ":" not in raw:
            return None, raw
        tool_name, query = raw.split(":", 1)
        return tool_name.strip().lower(), query.strip()

    def _invoke_tool(
        self,
        rollout_idx: int,
        tool_usage: List[int],
        tool_name: Optional[str],
        query: str,
    ) -> str:
        if not query:
            return "Tool invocation rejected: empty query."

        if not tool_name or tool_name not in self.allowed_tools:
            return "Tool invocation rejected: unknown tool."

        if tool_usage[rollout_idx] >= self.generation_config.max_tool_invocations:
            return "Tool invocation rejected: routing budget exhausted."

        if self.tool_invoker is None:
            return "Tool invocation rejected: tool unavailable."

        try:
            response = self.tool_invoker(query)
        except Exception as exc:  # noqa: BLE001
            response = f"Tool error: {exc}"

        tool_usage[rollout_idx] += 1
        return response

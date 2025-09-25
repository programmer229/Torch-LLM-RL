from __future__ import annotations

from typing import Optional

import torch

from SimpleTorchLLMRL.chat.message import Message, MessageType, Rollout
from SimpleTorchLLMRL.model.generate import ModelGenerate
from SimpleTorchLLMRL.router_r1.config import RouterToolConfig


class QwenToolInvoker:
    """Lazy loader around a local Qwen2.5 generation stack."""

    def __init__(self, config: RouterToolConfig) -> None:
        self.config = config
        self._generator: Optional[ModelGenerate] = None

    def _resolve_dtype(self, name: Optional[str]):
        if not name:
            return None
        if not hasattr(torch, name):
            raise ValueError(f"Unsupported torch dtype for tool model: {name}")
        return getattr(torch, name)

    def _ensure_ready(self) -> None:
        if self._generator is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer  # local import for faster CLI start

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}
        if self.config.device_map is not None:
            model_kwargs["device_map"] = self.config.device_map

        resolved_dtype = self._resolve_dtype(self.config.torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )

        self._generator = ModelGenerate(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            top_p=self.config.top_p,
        )

    def __call__(self, query: str) -> str:
        self._ensure_ready()

        rollout = Rollout()
        if self.config.system_prompt:
            rollout.add_messages(Message(self.config.system_prompt, MessageType.SYSTEM))
        rollout.add_messages(Message(query, MessageType.MESSAGE))

        response = self._generator.rollout_generate_response(rollout)
        return response.content

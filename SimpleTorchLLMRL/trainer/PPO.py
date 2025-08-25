from abc import ABC, abstractmethod
import copy
# import token  # unused; removed to avoid shadowing stdlib names
from typing import List
from collections import defaultdict

from .trainer import Trainer
from SimpleTorchLLMRL.chat.message import Rollout, MessageType
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticLLM(nn.Module):
    """
    Wraps a base LLM with a scalar value head that predicts a value per token.
    """
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.value_head = nn.Linear(model.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns values of shape [B, T] corresponding to each token position.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state         # [B, T, H]
        values = self.value_head(last_hidden_state).squeeze(-1)  # [B, T]
        return values



class PPO(Trainer):
    def __init__(self, model,
                 critic_model=None,
                 tokenizer=None,
                 eps: float = 0.2) -> None:
        super().__init__()
        self.model = model
        self.ref_model = copy.deepcopy(model).eval() 
        # use provided critic_model if passed; otherwise wrap a copy of the actor
        self.critic_model = critic_model if critic_model is not None else ActorCriticLLM(copy.deepcopy(model))
        self.tokenizer = tokenizer
        self.eps = eps

    def _calculate_advantage(self,
                             rewards: torch.Tensor,
                             input_ids: torch.Tensor,
                             attention_mask: torch.Tensor | None = None,
                             use_gae: bool = False) -> torch.Tensor:
        """
        Simple per-token advantage: broadcast per-rollout reward across sequence
        and subtract critic values. Shape: [B, T].
        (You can replace this with GAE later without touching the call sites.)
        """
        # Critic values per token: [B, T]
        with torch.no_grad():
            values = self.critic_model(input_ids=input_ids, attention_mask=attention_mask)

        B, T = input_ids.shape
        rewards = rewards.to(input_ids.device).view(B)  # [B]
        seq_rewards = rewards.view(B, 1).expand(B, T)   # [B, T]
        advantage = seq_rewards - values                # [B, T]

        # Optional normalization (helps stability)
        # mask out pads if attention_mask provided
        if attention_mask is not None:
            mask = attention_mask.bool()
            mean = advantage[mask].mean() if mask.any() else advantage.mean()
            std = advantage[mask].std().clamp_min(1e-8) if mask.any() else advantage.std().clamp_min(1e-8)
            advantage = (advantage - mean) / std
        else:
            std = advantage.std().clamp_min(1e-8)
            advantage = (advantage - advantage.mean()) / std

        return advantage

    def _calculate_prob_ratios(self,
                               input_ids: torch.Tensor,
                               attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        PPO ratios for taken actions only:
        ratio_t = exp(logπ(a_t|s_t) - logπ_ref(a_t|s_t)), shape [B, T].
        """
        # Current policy logits
        policy_out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = policy_out.logits  # [B, T, V]

        # Reference policy logits (no grad)
        with torch.no_grad():
            ref_out = self.ref_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            ref_logits = ref_out.logits  # [B, T, V]

        # Log-probs of actually taken tokens
        logp = F.log_softmax(logits, dim=-1).gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)      # [B, T]
        ref_logp = F.log_softmax(ref_logits, dim=-1).gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]

        ratios = torch.exp(logp - ref_logp)  # [B, T]
        return ratios

    def calculate_loss(self, rollouts: List[Rollout], rewards: torch.Tensor) -> torch.Tensor:
        """
        Computes PPO loss only on MODEL message tokens inside each rollout.
        """
        if len(rollouts) == 0:
            # Handle empty batch
            device = next(self.model.parameters()).device if self.model is not None else "cpu"
            return torch.tensor(0.0, device=device)

        device = next(self.model.parameters()).device

        # Build token ranges for MODEL messages (indices into each rollout's tokenized conversation)
        backprop_indices = defaultdict(list)
        for rollout_idx, rollout in enumerate(rollouts):
            token_pos = 0
            for msg in rollout:
                msg_tokens = self.tokenizer.encode(msg.content, add_special_tokens=False)
                if msg.type == MessageType.MODEL:
                    backprop_indices[rollout_idx].append((token_pos, token_pos + len(msg_tokens)))
                token_pos += len(msg_tokens)

        # Tokenize the full conversations (same template as above: raw concatenation of msg.content)
        formatted_strings = [rollout.format_conversation_str() for rollout in rollouts]
        encoded_inputs = [self.tokenizer.encode(s, add_special_tokens=False) for s in formatted_strings]

        # Handle empty rollouts case
        if not encoded_inputs:
            return torch.tensor(0.0, device=device)

        # Pad to same length
        max_length = max(len(seq) for seq in encoded_inputs)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be set for padding.")

        input_ids = torch.tensor(
            [seq + [pad_id] * (max_length - len(seq)) for seq in encoded_inputs],
            device=device,
            dtype=torch.long,
        )  # [B, T]
        attention_mask = (input_ids != pad_id).long()  # [B, T]

        # Per-token advantages (normalized), shape [B, T]
        advantage = self._calculate_advantage(rewards=rewards,
                                              input_ids=input_ids,
                                              attention_mask=attention_mask)

        # Ratios for taken tokens, shape [B, T]
        ratios = self._calculate_prob_ratios(input_ids=input_ids, attention_mask=attention_mask)

        # Compute PPO loss only over MODEL-message token spans
        rollout_losses = []
        for rollout_idx in range(len(rollouts)):
            model_message_losses = []
            if rollout_idx in backprop_indices:
                for start_pos, end_pos in backprop_indices[rollout_idx]:
                    # Slice spans (ignore pads beyond real length)
                    message_mask_len = min(end_pos, attention_mask.shape[1])
                    if start_pos >= message_mask_len:
                        continue

                    r = ratios[rollout_idx, start_pos:message_mask_len]  # [S]
                    adv = advantage[rollout_idx, start_pos:message_mask_len]  # [S]
                    if r.numel() == 0:
                        continue

                    clipped = torch.clamp(r, 1.0 - self.eps, 1.0 + self.eps)
                    loss_per_token = torch.min(r * adv, clipped * adv)  # [S]
                    message_loss = loss_per_token.mean()
                    model_message_losses.append(message_loss)

            if model_message_losses:
                rollout_loss = torch.stack(model_message_losses).mean()
                rollout_losses.append(rollout_loss)

        if rollout_losses:
            # Negative sign: we minimize, PPO maximizes objective
            loss = -torch.stack(rollout_losses).mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss

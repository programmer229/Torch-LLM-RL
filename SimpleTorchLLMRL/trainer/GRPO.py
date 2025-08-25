import copy
from typing import List
from collections import defaultdict

from .trainer import Trainer
from SimpleTorchLLMRL.chat.message import Rollout, MessageType
import torch
import torch.nn.functional as F

class GRPO(Trainer):
    def __init__(self, model, tokenizer, eps, use_kl=False) -> None:
        super().__init__()
        self.model = model
        self.ref_model = copy.deepcopy(model).eval()
        self.base_model = copy.deepcopy(model).eval()
        self.tokenizer = tokenizer
        self.eps = eps

    def _calculate_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        # rewards: [B]
        if rewards.numel() <= 1:
            return rewards.clamp(min=-1, max=1)
        r = rewards.float()
        std = r.std()
        if std <= 1e-8:
            return torch.zeros_like(r)
        adv = (r - r.mean()) / std
        return torch.nan_to_num(adv, nan=0.0)

    def _calculate_prob_ratios(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns PPO ratios for taken actions:
            ratios[b, t] = exp(logπ(a_{t+1}|s_t) - logπ_ref(a_{t+1}|s_t))
        Shape: [B, T-1]
        """
        # Actions are the next-token ids
        actions = input_ids[:, 1:]                        # [B, T-1]

        # Align logits to predict tokens at positions 1..T-1
        policy_out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = policy_out.logits[:, :-1, :]             # [B, T-1, V]

        with torch.no_grad():
            ref_out = self.ref_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            ref_logits = ref_out.logits[:, :-1, :]        # [B, T-1, V]

        logp = F.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)      # [B, T-1]
        ref_logp = F.log_softmax(ref_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        ratios = torch.exp(logp - ref_logp)               # [B, T-1]
        # Mask out any positions that were padding at t or t+1
        valid = (attention_mask[:, :-1] & attention_mask[:, 1:]).bool()
        ratios = ratios * valid.to(ratios.dtype)
        return ratios


    def calculate_kl(self, input_ids, base_model, model):

        with torch.no_grad():
            ref_out = base_model(input_ids=input_ids, return_dict=True)
            ref_logits = ref_out.logits
        
        model_logits = model(input_ids=input_ids, return_dict=True)
        model_probs = torch.nn.log_softmax(model_logits, dim=-1)
        
        ref_logits_probs = torch.nn.log_softmax(ref_logits, dim=-1)
        ref_logits_probs = torch.nn.log_softmax(ref_logits, dim=-1)



        pass

    def update_ref_model(self):
        self.ref_model.load_state_dict(self.model.state_dict())

    def calculate_loss(self, rollouts: List[Rollout], rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute GRPO/PPO-style loss on MODEL tokens only.
        """
        if len(rollouts) == 0:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)

        device = next(self.model.parameters()).device

        # --- Build input_ids and MODEL spans from the SAME tokenization path ---
        # We concatenate msg.content tokens ourselves to avoid mismatch with format_conversation_str().
        batch_token_lists = []
        model_spans = defaultdict(list)  # rollout_idx -> List[(start, end)] in TOKEN POSITIONS
        for b_idx, r in enumerate(rollouts):
            toks = []
            pos = 0
            for msg in r:
                # Tokenize content exactly as inserted into the conversation (no special tokens here)
                msg_ids = self.tokenizer.encode(msg.content, add_special_tokens=False)
                if msg.type == MessageType.MODEL and len(msg_ids) > 0:
                    model_spans[b_idx].append((pos, pos + len(msg_ids)))  # token positions of MODEL message
                toks.extend(msg_ids)
                pos += len(msg_ids)
            batch_token_lists.append(toks)

        # Handle empty batch safety
        if not batch_token_lists:
            return torch.tensor(0.0, device=device)

        # Pad to tensor
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be set for padding.")

        max_len = max(len(seq) for seq in batch_token_lists)
        input_ids = torch.full((len(batch_token_lists), max_len), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(batch_token_lists):
            if len(seq) > 0:
                input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask = (input_ids != pad_id).long()  # [B, T]

        # --- Ratios for actions (T-1) ---
        ratios = self._calculate_prob_ratios(input_ids=input_ids, attention_mask=attention_mask)  # [B, T-1]

        # --- Per-rollout scalar advantage (broadcast over its tokens) ---
        # rewards: [B] → normalize to [B]
        rewards = rewards.to(device).view(-1)
        advantage_scalar = self._calculate_advantage(rewards)  # [B]

        # --- Accumulate losses over MODEL-token spans (mapped to action indices) ---
        rollout_losses = []
        B, T = input_ids.shape
        for b in range(len(rollouts)):
            msg_losses = []
            if b in model_spans:
                for (start_tok, end_tok) in model_spans[b]:
                    # Map TOKEN positions (1..T-1) to ACTION index (0..T-2):
                    # token at k is predicted by action index k-1
                    a_start = max(start_tok - 1, 0)
                    a_end = max(end_tok - 1, 0)
                    if a_end <= a_start:
                        continue  # first token in a message has no preceding state to predict it

                    # Clip to available action length (T-1)
                    a_end = min(a_end, ratios.shape[1])
                    if a_end <= a_start:
                        continue

                    r_slice = ratios[b, a_start:a_end]  # [S]
                    if r_slice.numel() == 0:
                        continue

                    clipped = torch.clamp(r_slice, 1.0 - self.eps, 1.0 + self.eps)
                    adv = advantage_scalar[b]  # scalar for this rollout
                    # PPO clipped objective (maximize), so we take negative to minimize
                    loss_per_token = -torch.min(r_slice * adv, clipped * adv)  # [S]
                    msg_losses.append(loss_per_token.mean())

            if msg_losses:
                rollout_losses.append(torch.stack(msg_losses).mean())

        if not rollout_losses:
            return torch.tensor(0.0, device=device)

        return torch.stack(rollout_losses).mean()

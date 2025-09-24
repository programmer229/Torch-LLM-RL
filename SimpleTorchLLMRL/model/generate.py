import torch
from typing import List, Optional

from SimpleTorchLLMRL.chat.message import Rollout, Message, MessageType


class ModelGenerate():

    def __init__(self, model, tokenizer, max_new_tokens: int = 50, temperature: float = 0.7,
                 do_sample: bool = True, pad_token_id: Optional[int] = None,
                 top_p: Optional[float] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.pad_token_id = pad_token_id
        self.top_p = top_p
    
    def _generate(self, input_ids):
        """Generate text using Hugging Face model."""
        pad_token_id = self.pad_token_id if self.pad_token_id is not None else self.tokenizer.eos_token_id

        self.model.eval()

        with torch.no_grad():
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            generate_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": self.do_sample,
                "pad_token_id": pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.top_p is not None:
                generate_kwargs["top_p"] = self.top_p

            outputs = self.model.generate(**generate_kwargs)

            new_tokens = outputs[:, input_ids.shape[1]:]
            return new_tokens.cpu()

    def rollout_generate_response(self, rollout: Rollout) -> Message:
        # Just use the simple string format
        prompt = rollout.format_conversation_str()
        
        # Tokenize
        inputs_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate
        response_ids = self._generate(inputs_ids)
        
        # Decode - that's it!
        output = self.tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
        
        return Message(output, MessageType.MODEL)
    
    def batch_rollout_generate_response(self, rollouts: List[Rollout]):
        prompts = [rollout.format_conversation_str() for rollout in rollouts]
        
        inputs_ids = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )["input_ids"]

        responses_ids = self._generate(inputs_ids)
        outputs = [self.tokenizer.decode(response_ids, skip_special_tokens=True).strip() 
                  for response_ids in responses_ids]

        messages = [Message(output, MessageType.MODEL) for output in outputs]

        for message, rollout in zip(messages, rollouts):
            rollout.add_messages(message)
        




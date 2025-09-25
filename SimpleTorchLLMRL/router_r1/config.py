from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RouterDataConfig:
    """Configuration controlling data loading and prompt formatting."""

    system_prompt: str
    train_batch_size: int = 64
    val_batch_size: int = 64
    max_prompt_length: int = 4096
    max_response_length: int = 1024
    shuffle: bool = True


@dataclass
class RouterGenerationConfig:
    """Text generation hyper parameters for Router-R1 style rollouts."""

    max_new_tokens: int = 768
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    max_turns: int = 4
    max_tool_invocations: int = 4


@dataclass
class RouterToolConfig:
    """Configuration for the single routing tool (Qwen2.5 by default)."""

    enabled: bool = True
    tool_name: str = "qwen2.5"
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: Optional[str] = (
        "You are Qwen 2.5 acting as a specialist tool. "
        "Answer the provided sub-question concisely and factually."
    )
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = False
    device_map: Optional[str] = "auto"
    torch_dtype: Optional[str] = None


@dataclass
class RouterTrainerConfig:
    """High level training loop configuration."""

    total_epochs: int = 1
    total_training_steps: Optional[int] = None
    val_before_train: bool = False
    val_only: bool = False
    test_freq: int = 0
    save_freq: int = 0
    critic_warmup: int = 0
    project_name: Optional[str] = None
    experiment_name: Optional[str] = None
    logger_backends: List[str] = field(default_factory=lambda: ["console"])
    gradient_accumulation: int = 1
    max_grad_norm: Optional[float] = 1.0
    log_train_completions: bool = False
    log_train_completions_file: Optional[str] = None


@dataclass
class RouterRewardConfig:
    """Parameters controlling Router-R1 reward shaping."""

    reward_metric: str = "em"
    cost_coefficient: float = 0.0
    format_penalty_small: float = -0.25
    format_penalty_medium: float = -0.5
    format_penalty_large: float = -1.0
    num_examine: int = 0


@dataclass
class RouterValidationConfig:
    """Validation loop configuration."""

    max_batches: Optional[int] = None
    store_completions: bool = False


@dataclass
class RouterTrainingConfig:
    """Convenience bundle collecting all Router specific sub-configs."""

    data: RouterDataConfig
    generation: RouterGenerationConfig = field(default_factory=RouterGenerationConfig)
    trainer: RouterTrainerConfig = field(default_factory=RouterTrainerConfig)
    reward: RouterRewardConfig = field(default_factory=RouterRewardConfig)
    validation: RouterValidationConfig = field(default_factory=RouterValidationConfig)
    tool: RouterToolConfig = field(default_factory=RouterToolConfig)

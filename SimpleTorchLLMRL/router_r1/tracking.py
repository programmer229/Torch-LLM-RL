from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


class _ConsoleLogger:
    def log(self, data: Dict[str, float], step: int) -> None:
        formatted = ", ".join(f"{key}={value:.4f}" for key, value in sorted(data.items()))
        print(f"[step={step}] {formatted}")


@dataclass
class Tracking:
    """Lightweight multi-backend tracker inspired by Router-R1's utility."""

    project_name: Optional[str]
    experiment_name: Optional[str]
    default_backends: Iterable[str] = ("console",)
    config: Optional[dict] = None

    def __post_init__(self) -> None:
        self._loggers: Dict[str, object] = {}
        backends = list(self.default_backends)
        for backend in backends:
            if backend not in {"console", "wandb"}:
                raise ValueError(f"Unsupported tracking backend: {backend}")

        if "console" in backends:
            self._loggers["console"] = _ConsoleLogger()

        if "wandb" in backends:
            try:
                import wandb  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("wandb backend requested but wandb is not installed") from exc

            wandb.init(project=self.project_name, name=self.experiment_name, config=self.config)
            self._loggers["wandb"] = wandb

    def log(self, data: Dict[str, float], step: int, *, backend: Optional[str] = None) -> None:
        if backend:
            targets = [backend]
        else:
            targets = list(self._loggers.keys())

        for name in targets:
            logger = self._loggers.get(name)
            if not logger:
                continue
            if name == "wandb":
                logger.log(data=data, step=step)
            else:
                logger.log(data=data, step=step)

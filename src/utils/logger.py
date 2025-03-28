import os
from typing import Any, Dict, Optional

import wandb


class WandbLogger:
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "online",
    ):
        """Initialize Weights & Biases logger.

        Args:
            project: Project name in W&B
            name: Run name. If None, W&B will generate a random name
            config: Configuration dictionary to log
            mode: W&B mode ("online", "offline", "disabled")
        """
        self.project = project
        self.name = name
        self.config = config or {}
        self.mode = mode

        # Initialize wandb
        wandb.init(
            project=project,
            name=name,
            config=self.config,
            mode=mode,
            reinit=True,
        )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Step number for x-axis
        """
        wandb.log(metrics, step=step)

    def log_model(self, model: Any, name: str = "model"):
        """Log model architecture to W&B.

        Args:
            model: PyTorch model
            name: Name of the model in W&B
        """
        wandb.watch(model, name=name)

    def log_config(self, config: Dict[str, Any]):
        """Log configuration to W&B.

        Args:
            config: Configuration dictionary
        """
        wandb.config.update(config)

    def log_media(self, media_dict: Dict[str, Any], step: Optional[int] = None):
        """Log media (images, videos, etc.) to W&B.

        Args:
            media_dict: Dictionary of media names and objects
            step: Step number for x-axis
        """
        wandb.log(media_dict, step=step)

    def log_artifact(self, artifact_path: str, name: str, type: str):
        """Log artifact to W&B.

        Args:
            artifact_path: Path to the artifact file
            name: Name of the artifact
            type: Type of the artifact (e.g., "model", "dataset")
        """
        artifact = wandb.Artifact(name=name, type=type)
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def finish(self):
        """Close W&B run."""
        wandb.finish()
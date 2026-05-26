"""Simple experiment logging utilities."""

import json
from pathlib import Path
from datetime import datetime


class ExperimentLogger:
    """Logs experiment configuration and metrics."""

    def __init__(self, log_dir: str, run_name: str | None = None):
        """
        Args:
            log_dir: Base directory for all logs
            run_name: Optional run name (default: timestamp)
        """
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = Path(log_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = []
        print(f"Logging to: {self.run_dir}")

    def save_config(self, config: dict):
        """Save configuration to JSON."""
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to: {config_path}")

    def log_metrics(self, epoch: int, step: int, metrics: dict):
        """Log metrics for a training step."""
        entry = {
            "epoch": epoch,
            "step": step,
            **metrics,
        }
        self.metrics.append(entry)

    def save_metrics(self):
        """Save all metrics to JSON."""
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

"""Utility functions: seeding, config loading, logging."""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML config, merging with base config if specified."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Merge with base config if it exists
    if "base" in cfg:
        base_path = Path(config_path).parent / cfg.pop("base")
        with open(base_path, "r") as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update(cfg)
        cfg = base_cfg

    return cfg


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

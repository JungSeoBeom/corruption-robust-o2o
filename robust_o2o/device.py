from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def resolve_device(requested: str = "auto", cuda_device: int = 0) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{cuda_device}")
            torch.cuda.set_device(cuda_device)
            return device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        requested = f"cuda:{cuda_device}"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        device = torch.device(requested)
        torch.cuda.set_device(device)
        return device
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested != "cpu":
        raise ValueError(f"Unsupported device {requested!r}")
    return torch.device("cpu")


def seed_everything(seed: int, env: Optional[object] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        if hasattr(env, "seed"):
            env.seed(seed)
        action_space = getattr(env, "action_space", None)
        if action_space is not None and hasattr(action_space, "seed"):
            action_space.seed(seed)


def clear_accelerator_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

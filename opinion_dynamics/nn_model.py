# opinion_dynamics/nn_model.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpinionUpdateNet(nn.Module):
    """
    Small MLP that maps psychological traits + authority context
    to an effective influence strength in [0, 1].

    Expected input per interaction (5-dim vector):

        x = [
            is_authority_flag,   # 0/1, NOT standardized
            vigilance_score,     # standardized
            media_literacy,      # standardized
            trust_authority,     # standardized
            fam_academic_score,  # standardized
        ]

    Normalization stats (X_mean, X_std) are attached to the model
    when loaded from a checkpoint produced by train_nn_real.py.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Filled when loading a trained checkpoint
        self.X_mean: Optional[torch.Tensor] = None
        self.X_std: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Squash to (0,1)
        return torch.sigmoid(self.fc3(x))


@dataclass
class NNConfig:
    input_dim: int = 5
    hidden_dim: int = 32
    device: str = "cpu"
    model_path: Optional[str] = None


def build_default_model(cfg: Optional[NNConfig] = None) -> OpinionUpdateNet:
    """
    Create an untrained OpinionUpdateNet (random weights).
    Used as a fallback if no checkpoint exists.
    """
    if cfg is None:
        cfg = NNConfig()

    device = torch.device(cfg.device)
    model = OpinionUpdateNet(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    model.X_mean = None
    model.X_std = None
    model.eval()
    return model


def load_or_init_model(cfg: Optional[NNConfig] = None) -> OpinionUpdateNet:
    """
    Load a trained OpinionUpdateNet if cfg.model_path exists.
    Otherwise return a randomly initialized model.

    Supported checkpoint formats:

    1) New format (from train_nn_real.py):
        {
            "state_dict": model_state_dict,
            "X_mean": np.ndarray,
            "X_std": np.ndarray,
        }

    2) Legacy format:
        model_state_dict
    """
    if cfg is None:
        cfg = NNConfig()

    device = torch.device(cfg.device)

    if cfg.model_path is not None and Path(cfg.model_path).is_file():
        model = OpinionUpdateNet(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(device)

        # PyTorch ≥2.6: must explicitly allow non-weights objects
        ckpt = torch.load(
            cfg.model_path,
            map_location=device,
            weights_only=False,
        )

        X_mean = None
        X_std = None

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            # New checkpoint format
            model.load_state_dict(ckpt["state_dict"])
            X_mean = ckpt.get("X_mean", None)
            X_std = ckpt.get("X_std", None)
        else:
            # Old format: plain state_dict
            model.load_state_dict(ckpt)

        if X_mean is not None and X_std is not None:
            model.X_mean = torch.as_tensor(
                X_mean, dtype=torch.float32, device=device
            )
            model.X_std = torch.as_tensor(
                X_std, dtype=torch.float32, device=device
            )
        else:
            model.X_mean = None
            model.X_std = None

        model.eval()
        return model

    # Fallback
    return build_default_model(cfg)

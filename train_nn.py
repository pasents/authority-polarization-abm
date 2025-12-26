import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from opinion_dynamics.nn_model import OpinionUpdateNet, NNConfig


# ---------- CONFIG ----------

N_SAMPLES = 200_000       # training samples
BATCH_SIZE = 512
EPOCHS = 20
LR = 1e-3

MU = 0.3                  # must match your SimulationParams.mu
AUTHORITY_BOOST = 0.2     # must match dynamics.update_opinions default
SEED = 42

MODEL_SAVE_PATH = Path("opinion_dynamics/trained_nn.pt")


# ---------- DATA GENERATION ----------

def generate_training_data(
    n_samples: int,
    mu: float = MU,
    authority_boost: float = AUTHORITY_BOOST,
    seed: int = SEED,
):
    """
    Generate synthetic interactions that follow your analytic rule.
    We then train the NN to approximate that rule.

    Features per sample:
      x = [opinion_i, opinion_j, vigilance_j,
           share_propensity_i, is_authority_i]

    Target:
      y = mu_eff / mu  in [0, 1]
      where mu_eff = mu*(1 - vigilance_j) + authority_boost * is_authority_i
    """
    rng = np.random.default_rng(seed)

    # Opinions in [-1, 1]
    oi = rng.uniform(-1.0, 1.0, size=n_samples)
    oj = rng.uniform(-1.0, 1.0, size=n_samples)

    # Authority flag (0 = neutral, 1 = authority)
    is_auth = rng.integers(0, 2, size=n_samples).astype(float)

    # Vigilance: match thesis means (neutral ~0.45, authority ~0.35)
    vig_neutral = np.clip(rng.normal(0.45, 0.15, size=n_samples), 0.0, 1.0)
    vig_auth = np.clip(rng.normal(0.35, 0.15, size=n_samples), 0.0, 1.0)
    vigilance = np.where(is_auth == 1.0, vig_auth, vig_neutral)

    # Share propensity: match thesis means (neutral ~0.22, authority ~0.34)
    share_neutral = np.clip(rng.normal(0.22, 0.15, size=n_samples), 0.0, 1.0)
    share_auth = np.clip(rng.normal(0.34, 0.15, size=n_samples), 0.0, 1.0)
    share_prop = np.where(is_auth == 1.0, share_auth, share_neutral)

    # Analytic rule for effective influence
    mu_eff = mu * (1.0 - vigilance) + authority_boost * is_auth
    mu_eff = np.clip(mu_eff, 0.0, 1.0)

    # Target is base factor in [0, 1], since in the ABM we do mu_eff = mu * nn_out
    y = np.clip(mu_eff / mu, 0.0, 1.0)

    # Features
    X = np.stack([oi, oj, vigilance, share_prop, is_auth], axis=1).astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)

    return X, y


# ---------- TRAINING LOOP ----------

def set_global_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train():
    set_global_seed(SEED)

    print("Generating training data...")
    X, y = generate_training_data(N_SAMPLES)

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    cfg = NNConfig(device="cpu", input_dim=5, hidden_dim=32, model_path=None)
    device = torch.device(cfg.device)

    model = OpinionUpdateNet(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)          # in (0,1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch:02d}/{EPOCHS} - MSE loss: {epoch_loss:.6f}")

    # Save trained weights
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nSaved trained model to: {MODEL_SAVE_PATH.resolve()}")


if __name__ == "__main__":
    train()

# train_nn_real.py

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from opinion_dynamics.nn_model import OpinionUpdateNet, NNConfig


# ========= CONFIG =========

DATA_PATH = Path("data/thesis_clean.csv")   # CSV exported from Stata
MODEL_SAVE_PATH = Path("opinion_dynamics/trained_nn_real.pt")

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
SEED = 42


# ========= COLUMN NAMES (match your CSV) =========

COL_SHARE = "share_score"
COL_VIG = "vigilance_score"
COL_MEDIA_LIT = "media_literacy"
COL_TRUST_AUTH = "trust_authority"
COL_FAM_ACAD = "fam_academic_score"
COL_CONTEXT_AUTH = "context_authority"      # "Authoritative setting" / "Neutral setting"


# ========= UTILITIES =========

def set_global_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# Likert mappings
SHARE_MAP = {
    "Extremely unlikely": 0.0,
    "Unlikely": 0.25,
    "Neutral": 0.5,
    "Likely": 0.75,
    "Extremely likely": 1.0,
}

VIG_MAP = {
    "Not at all vigilant": 1.0,
    "Slightly vigilant": 2.0,
    "Moderately vigilant": 3.0,
    "Very vigilant": 4.0,
    "Extremely vigilant": 5.0,
}

LIKERT5_AGREE = {
    "Strongly disagree": 1.0,
    "Disagree": 2.0,
    "Neutral": 3.0,
    "Agree": 4.0,
    "Strongly agree": 5.0,
}

FAM_MAP = {
    "Not at all familiar": 1.0,
    "Slightly familiar": 2.0,
    "Moderately familiar": 3.0,
    "Very familiar": 4.0,
    "Extremely familiar": 5.0,
}


def to_float_mixed(series: pd.Series, label_map: dict | None = None) -> pd.Series:
    """Convert a column that is a mix of numbers and text labels into floats."""
    def conv(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x)
        if label_map and s in label_map:
            return float(label_map[s])
        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.apply(conv)


def load_and_prepare_data() -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Load thesis_clean.csv, build numeric features and target.

    X = [is_authority_flag, vigilance_num, media_num, trust_num, fam_num]
    y = share_num in [0, 1]
    """
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find data file at {DATA_PATH}. "
            f"Make sure thesis_clean.csv is in data/."
        )

    # ==== 1. LOAD CSV ====
    df = pd.read_csv(DATA_PATH)
    print("Loaded CSV shape:", df.shape)

    # ==== 2. BUILD NUMERIC TARGET ====
    if COL_SHARE not in df.columns:
        raise KeyError(f"{COL_SHARE!r} not found in CSV columns: {list(df.columns)}")

    df["share_num"] = df[COL_SHARE].map(SHARE_MAP)

    if df["share_num"].isna().any():
        missing_vals = df[COL_SHARE][df["share_num"].isna()].unique()
        raise ValueError(
            f"Some share_score values were not mapped: {missing_vals}. "
            f"Update SHARE_MAP if needed."
        )

    # ==== 3. BUILD NUMERIC FEATURES ====

    # Vigilance
    if COL_VIG not in df.columns:
        raise KeyError(f"{COL_VIG!r} not found in CSV.")
    df["vigilance_num"] = df[COL_VIG].map(VIG_MAP)
    if df["vigilance_num"].isna().any():
        missing_vals = df[COL_VIG][df["vigilance_num"].isna()].unique()
        raise ValueError(
            f"Some vigilance_score values were not mapped: {missing_vals}. "
            f"Update VIG_MAP if needed."
        )

    # Trust & media literacy
    if COL_TRUST_AUTH not in df.columns or COL_MEDIA_LIT not in df.columns:
        raise KeyError("trust_authority or media_literacy missing from CSV.")
    df["trust_num"] = to_float_mixed(df[COL_TRUST_AUTH], LIKERT5_AGREE)
    df["media_num"] = to_float_mixed(df[COL_MEDIA_LIT], LIKERT5_AGREE)

    # Familiarity
    if COL_FAM_ACAD not in df.columns:
        raise KeyError(f"{COL_FAM_ACAD!r} not found in CSV.")
    df["fam_num"] = df[COL_FAM_ACAD].map(FAM_MAP)

    if df[["trust_num", "media_num", "fam_num"]].isna().any().any():
        raise ValueError(
            "NaNs found in trust_num/media_num/fam_num after mapping. "
            "Check label maps."
        )

    # Authority flag from context_authority
    if COL_CONTEXT_AUTH not in df.columns:
        raise KeyError(f"{COL_CONTEXT_AUTH!r} not found in CSV.")
    df["is_authority_flag"] = (
        df[COL_CONTEXT_AUTH] == "Authoritative setting"
    ).astype(float)

    # ==== 4. BUILD FINAL MATRICES ====
    X_cols = [
        "is_authority_flag",
        "vigilance_num",
        "media_num",
        "trust_num",
        "fam_num",
    ]

    df_clean = df[X_cols + ["share_num"]].copy()

    X = df_clean[X_cols].to_numpy(dtype=np.float32)
    y = df_clean[["share_num"]].to_numpy(dtype=np.float32)

    print("Rows before NaN mask:", X.shape[0])

    mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y).any(axis=1))
    X = X[mask]
    y = y[mask]

    print("Rows after NaN mask:", X.shape[0])

    if X.shape[0] == 0:
        raise ValueError("No rows left after cleaning; check mappings and CSV values.")

    # ==== 5. STANDARDIZE FEATURES (EXCEPT BINARY FLAG) ====
    X_mean = X[:, 1:].mean(axis=0, keepdims=True).astype(np.float32)
    X_std = X[:, 1:].std(axis=0, keepdims=True).astype(np.float32)
    X[:, 1:] = (X[:, 1:] - X_mean) / (X_std + 1e-8)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    return X_tensor, y_tensor, X_mean, X_std


# ========= TRAINING =========

def train():
    set_global_seed(SEED)

    print(f"Loading data from {DATA_PATH}...")
    X, y, X_mean, X_std = load_and_prepare_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    cfg = NNConfig(
        input_dim=5,   # is_authority + 4 traits
        hidden_dim=32,
        device="cpu",
        model_path=None,
    )
    device = torch.device(cfg.device)

    model = OpinionUpdateNet(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Starting training on real participant data...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)          # in (0, 1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch:02d}/{EPOCHS} - MSE loss: {epoch_loss:.6f}")

    # ========= EVALUATION METRICS ON FULL DATA =========
    model.eval()
    with torch.no_grad():
        preds = model(X.to(device)).cpu().numpy().flatten()
        y_true = y.numpy().flatten()

    mse = float(((preds - y_true) ** 2).mean())
    rmse = float(np.sqrt(mse))
    mae = float(np.abs(preds - y_true).mean())
    # R^2
    ss_res = float(((y_true - preds) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("\n=== Train metrics on full dataset ===")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R^2  : {r2:.4f}")

    # ===== SAVE MODEL + NORMALIZATION =====
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"\nSaved real-data-trained model to: {MODEL_SAVE_PATH.resolve()}")


if __name__ == "__main__":
    train()

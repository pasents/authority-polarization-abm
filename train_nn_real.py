# train_nn_real.py
# Robust, defensible training script for thesis_clean.csv
# - Normalizes strings (strip)
# - Handles mixed numeric/text columns safely
# - Applies explicit quality filters (configurable; defaults are conservative)
# - Uses a real train/validation split
# - Saves model + normalization stats

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from opinion_dynamics.nn_model import OpinionUpdateNet, NNConfig

# ========= CONFIG =========

DATA_PATH = Path("data/thesis_clean.csv")
MODEL_SAVE_PATH = Path("opinion_dynamics/trained_nn_real.pt")

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
SEED = 42

# Holdout split
VAL_FRAC = 0.20

# Quality filters (set to None to disable a filter)
RECAPTCHA_MIN = 0.50        # typical: 0.5; set None to disable
ATTENDANCECHECK_MIN = 3     # your CSV seems 0–3; set None to disable
REQUIRE_FINISHED_TRUE = True

# ========= COLUMN NAMES (match your CSV) =========

COL_SHARE = "share_score"
COL_VIG = "vigilance_score"
COL_MEDIA_LIT = "media_literacy"
COL_TRUST_AUTH = "trust_authority"
COL_FAM_ACAD = "fam_academic_score"
COL_CONTEXT_AUTH = "context_authority"  # "Authoritative setting" / "Neutral setting"

# ========= MAPPINGS =========

SHARE_MAP: Dict[str, float] = {
    "Extremely unlikely": 0.0,
    "Unlikely": 0.25,
    "Neutral": 0.5,
    "Likely": 0.75,
    "Extremely likely": 1.0,
}

VIG_MAP: Dict[str, float] = {
    "Not at all vigilant": 1.0,
    "Slightly vigilant": 2.0,
    "Moderately vigilant": 3.0,
    "Very vigilant": 4.0,
    "Extremely vigilant": 5.0,
}

LIKERT5_AGREE: Dict[str, float] = {
    "Strongly disagree": 1.0,
    "Disagree": 2.0,
    "Neutral": 3.0,
    "Agree": 4.0,
    "Strongly agree": 5.0,
}

FAM_MAP: Dict[str, float] = {
    "Not at all familiar": 1.0,
    "Slightly familiar": 2.0,
    "Moderately familiar": 3.0,
    "Very familiar": 4.0,
    "Extremely familiar": 5.0,
}


# ========= UTILITIES =========

def set_global_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Determinism (CPU-safe; CUDA flags harmless if not used)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _norm_str(x) -> str:
    return str(x).strip()


def _norm_bool(x) -> bool:
    return _norm_str(x).upper() in {"TRUE", "1", "YES", "Y", "T"}


def to_float_mixed(series: pd.Series, label_map: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Convert a column that may contain:
      - floats/ints
      - numeric strings ("3", "4.2")
      - label strings (mapped via label_map)
    into floats. Unknowns => NaN.
    """
    def conv(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = _norm_str(x)
        if label_map is not None and s in label_map:
            return float(label_map[s])
        # numeric string?
        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.apply(conv)


def assert_no_unmapped(original: pd.Series, numeric: pd.Series, colname: str) -> None:
    """Fail fast if mapping/coercion produced NaNs for non-missing original values."""
    bad = numeric.isna() & (~original.isna())
    if bad.any():
        missing_vals = pd.Series(original[bad].astype(str).map(_norm_str).unique()).sort_values().to_list()
        raise ValueError(
            f"[ERROR] Unmapped/unparseable values found in {colname}: {missing_vals}. "
            f"Fix your label map or normalize the source values."
        )


def load_and_prepare_data() -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    X = [is_authority_flag, vigilance_num, media_num, trust_num, fam_num]
    y = share_num in [0,1]
    Returns:
      X_tensor, y_tensor, X_mean, X_std  (mean/std over columns 1: i.e., traits only)
    """
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Place thesis_clean.csv in data/.")

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded CSV shape: {df.shape}")

    # ---- Basic column presence checks ----
    required = [COL_SHARE, COL_VIG, COL_MEDIA_LIT, COL_TRUST_AUTH, COL_FAM_ACAD, COL_CONTEXT_AUTH]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"[ERROR] Missing required columns: {missing}")

    # ---- Quality filters (configurable) ----
    n0 = len(df)

    if REQUIRE_FINISHED_TRUE and "finished" in df.columns:
        df = df[df["finished"].apply(_norm_bool)]
        print(f"[FILTER] finished==TRUE: {n0} -> {len(df)}")
        n0 = len(df)

    if ATTENDANCECHECK_MIN is not None and "attendancecheck" in df.columns:
        df["attendancecheck_num"] = pd.to_numeric(df["attendancecheck"], errors="coerce")
        df = df[df["attendancecheck_num"] >= float(ATTENDANCECHECK_MIN)]
        print(f"[FILTER] attendancecheck>={ATTENDANCECHECK_MIN}: {n0} -> {len(df)}")
        n0 = len(df)

    if RECAPTCHA_MIN is not None and "q_recaptchascore" in df.columns:
        df["recaptcha_num"] = pd.to_numeric(df["q_recaptchascore"], errors="coerce")
        df = df[df["recaptcha_num"] >= float(RECAPTCHA_MIN)]
        print(f"[FILTER] q_recaptchascore>={RECAPTCHA_MIN}: {n0} -> {len(df)}")
        n0 = len(df)

    if len(df) == 0:
        raise ValueError("[ERROR] No rows left after quality filters. Relax thresholds or inspect data.")

    # ---- Map / coerce target and features ----
    # Target
    df["share_num"] = to_float_mixed(df[COL_SHARE], SHARE_MAP)
    assert_no_unmapped(df[COL_SHARE], df["share_num"], COL_SHARE)

    # Vigilance
    df["vigilance_num"] = to_float_mixed(df[COL_VIG], VIG_MAP)
    assert_no_unmapped(df[COL_VIG], df["vigilance_num"], COL_VIG)

    # Trust and media literacy (allow mixed numeric + Agree labels)
    df["trust_num"] = to_float_mixed(df[COL_TRUST_AUTH], LIKERT5_AGREE)
    df["media_num"] = to_float_mixed(df[COL_MEDIA_LIT], LIKERT5_AGREE)
    assert_no_unmapped(df[COL_TRUST_AUTH], df["trust_num"], COL_TRUST_AUTH)
    assert_no_unmapped(df[COL_MEDIA_LIT], df["media_num"], COL_MEDIA_LIT)

    # Familiarity
    df["fam_num"] = to_float_mixed(df[COL_FAM_ACAD], FAM_MAP)
    assert_no_unmapped(df[COL_FAM_ACAD], df["fam_num"], COL_FAM_ACAD)

    # Authority flag
    df["context_norm"] = df[COL_CONTEXT_AUTH].astype(str).map(_norm_str)
    allowed_context = {"Authoritative setting", "Neutral setting"}
    bad_ctx = ~df["context_norm"].isin(allowed_context)
    if bad_ctx.any():
        bad_vals = sorted(df.loc[bad_ctx, "context_norm"].unique().tolist())
        raise ValueError(f"[ERROR] Unexpected context_authority values: {bad_vals}")

    df["is_authority_flag"] = (df["context_norm"] == "Authoritative setting").astype(np.float32)

    # ---- Build final matrices ----
    X_cols = ["is_authority_flag", "vigilance_num", "media_num", "trust_num", "fam_num"]
    y_col = "share_num"

    df_model = df[X_cols + [y_col]].copy()

    # Drop any remaining NaNs (should be none unless you intentionally allow missing)
    n_before = len(df_model)
    df_model = df_model.dropna()
    print(f"[INFO] Rows after dropna: {n_before} -> {len(df_model)}")

    if len(df_model) < 10:
        raise ValueError("[ERROR] Too few rows after cleaning. Your filters/mappings are too strict or data is messy.")

    X = df_model[X_cols].to_numpy(dtype=np.float32)
    y = df_model[[y_col]].to_numpy(dtype=np.float32)

    # ---- Standardize traits only (cols 1:) ----
    X_mean = X[:, 1:].mean(axis=0, keepdims=True).astype(np.float32)
    X_std = X[:, 1:].std(axis=0, keepdims=True).astype(np.float32)
    X[:, 1:] = (X[:, 1:] - X_mean) / (X_std + 1e-8)

    return torch.from_numpy(X), torch.from_numpy(y), X_mean, X_std


def compute_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    y_true = y_true.flatten()
    preds = preds.flatten()

    mse = float(((preds - y_true) ** 2).mean())
    rmse = float(np.sqrt(mse))
    mae = float(np.abs(preds - y_true).mean())

    ss_res = float(((y_true - preds) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# ========= TRAINING =========

def train() -> None:
    set_global_seed(SEED)

    print(f"[INFO] Loading data from {DATA_PATH}...")
    X, y, X_mean, X_std = load_and_prepare_data()
    print(f"[INFO] Data ready: X={tuple(X.shape)}, y={tuple(y.shape)}")

    # ---- Train/val split ----
    n = X.shape[0]
    rng = np.random.default_rng(SEED)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_val = max(1, int(round(n * VAL_FRAC)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"[INFO] Split: train={len(train_idx)}, val={len(val_idx)}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    cfg = NNConfig(input_dim=5, hidden_dim=32, device="cpu", model_path=None)
    device = torch.device(cfg.device)

    model = OpinionUpdateNet(input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_state = None

    print("[INFO] Starting training on participant data...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)  # expected in (0,1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_idx)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(val_idx)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d}/{EPOCHS} - train_MSE={train_loss:.6f} val_MSE={val_loss:.6f}")

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Report metrics (train + val) ----
    model.eval()
    with torch.no_grad():
        preds_train = model(X_train.to(device)).cpu().numpy()
        preds_val = model(X_val.to(device)).cpu().numpy()

    m_train = compute_metrics(y_train.numpy(), preds_train)
    m_val = compute_metrics(y_val.numpy(), preds_val)

    print("\n=== Metrics (best validation model) ===")
    print(f"TRAIN: MSE={m_train['MSE']:.4f} RMSE={m_train['RMSE']:.4f} MAE={m_train['MAE']:.4f} R2={m_train['R2']:.4f}")
    print(f"VAL  : MSE={m_val['MSE']:.4f} RMSE={m_val['RMSE']:.4f} MAE={m_val['MAE']:.4f} R2={m_val['R2']:.4f}")

    # ---- Save model + normalization ----
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict(), "X_mean": X_mean, "X_std": X_std}
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"\n[INFO] Saved model to: {MODEL_SAVE_PATH.resolve()}")


if __name__ == "__main__":
    train()

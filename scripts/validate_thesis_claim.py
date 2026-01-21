from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

RAW_CLEAN = Path("data/thesis_clean.csv")  # adjust if needed

def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    sp = np.sqrt(((len(x)-1)*vx + (len(y)-1)*vy) / (len(x)+len(y)-2))
    if sp == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / sp

def _perm_test_diff_in_means(x: np.ndarray, y: np.ndarray, n_perm: int = 50_000, seed: int = 0) -> float:
    """
    Two-sided permutation test on mean difference (x - y).
    No scipy required.
    """
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan

    obs = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)

    more_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        x_p = pooled[:n_x]
        y_p = pooled[n_x:]
        stat = float(np.mean(x_p) - np.mean(y_p))
        if abs(stat) >= abs(obs):
            more_extreme += 1

    # add-one smoothing
    return (more_extreme + 1) / (n_perm + 1)

def main():
    if not RAW_CLEAN.exists():
        raise FileNotFoundError(f"Missing: {RAW_CLEAN.resolve()}")

    df = pd.read_csv(RAW_CLEAN)

    # --- Identify your key columns robustly
    # authority context flag
    if "context_authority" in df.columns:
        auth = df["context_authority"].astype("string").str.contains("Authoritative", na=False).astype(int)
    elif "condition" in df.columns:
        auth = df["condition"].astype("string").str.contains("Authoritative", na=False).astype(int)
    else:
        raise ValueError("Could not find context_authority or condition column to define authority flag.")

    # truth condition (False statements only)
    # Prefer is_true_condition if present
    if "is_true_condition" in df.columns:
        is_true = df["is_true_condition"].astype("string").str.contains("True", na=False)
    elif "condition" in df.columns:
        is_true = df["condition"].astype("string").str.startswith("True", na=False)
    else:
        raise ValueError("Could not infer truth condition (need is_true_condition or condition).")

    # sharing outcome: prefer numeric share_score; else map 1-5 Likert in share_willingness
    if "share_score" in df.columns:
        share = pd.to_numeric(df["share_score"], errors="coerce").to_numpy(dtype=float)
        share_label = "share_score"
    elif "share_willingness" in df.columns:
        share = pd.to_numeric(df["share_willingness"], errors="coerce").to_numpy(dtype=float)
        share_label = "share_willingness (numeric)"
    else:
        raise ValueError("Could not find share_score or share_willingness.")

    # --- Focus on false statements only (your thesis claim)
    mask_false = ~is_true
    x_auth = share[(mask_false) & (auth == 1)]
    x_neut = share[(mask_false) & (auth == 0)]

    # basic stats
    def _summ(x):
        x = x[~np.isnan(x)]
        return dict(n=int(len(x)), mean=float(np.mean(x)) if len(x) else np.nan, std=float(np.std(x, ddof=1)) if len(x) > 1 else np.nan)

    s_auth = _summ(x_auth)
    s_neut = _summ(x_neut)

    d = _cohen_d(x_auth, x_neut)
    p_perm = _perm_test_diff_in_means(x_auth, x_neut, n_perm=50_000, seed=1)

    print("\n=== Thesis-claim check: False statements only ===")
    print(f"Outcome: {share_label}")
    print(f"Neutral: n={s_neut['n']} mean={s_neut['mean']:.4f} sd={s_neut['std']:.4f}")
    print(f"Auth:    n={s_auth['n']} mean={s_auth['mean']:.4f} sd={s_auth['std']:.4f}")
    print(f"Mean diff (Auth - Neutral) = {(s_auth['mean'] - s_neut['mean']):.4f}")
    print(f"Cohen's d (Auth - Neutral) = {d:.4f}")
    print(f"Permutation p-value (two-sided, 50k) = {p_perm:.6f}")

    # --- Optional: quick OLS with robust SE (if statsmodels available)
    # This is NOT ordinal, but it’s a fast sanity check that direction is consistent.
    try:
        import statsmodels.api as sm

        tmp = df.loc[mask_false, :].copy()
        tmp["auth"] = auth[mask_false].to_numpy()

        y = pd.to_numeric(tmp["share_score"], errors="coerce")
        X = sm.add_constant(tmp[["auth"]])

        m = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")
        print("\n=== OLS sanity check (False only), robust SE ===")
        print(m.summary().as_text())
    except Exception as e:
        print("\n[INFO] statsmodels not available or failed; skipped OLS sanity check.")
        print(f"[INFO] Reason: {e}")

    print("\n=== Route A ABM sanity checklist ===")
    print("1) In Route A, update_opinions() must NOT add any authority_boost.")
    print("2) Route A treatment must ONLY enter via share_propensity.")
    print("3) Ensure both conditions use the SAME initialized graph + traits + seed.")
    print("4) Confirm share_propensity distributions differ by authority in your diagnostics.")

if __name__ == "__main__":
    main()

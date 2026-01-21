# scripts/clean_thesis_data.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# =========================
# INPUT / OUTPUT
# =========================
RAW_PATH = Path("data/thesis_raw.csv")          # raw export
OUT_CLEAN = Path("data/thesis_clean.csv")      # cleaned output
OUT_NOISE = Path("data/noise_report.csv")      # noise / unmapped values

# =========================
# NOISE HANDLING (approval gate)
# =========================
# If True: when ANY noise is detected, the script will write noise_report.csv and STOP
# without writing thesis_clean.csv. Rerun with APPROVE_NOISE=True to proceed.
REQUIRE_NOISE_APPROVAL = True
APPROVE_NOISE = False

# =========================
# QUALITY FILTERS (conservative defaults)
# =========================
FILTER_REQUIRE_FINISHED = True
FILTER_RECAPTCHA_MIN: float | None = 0.50
FILTER_ATTENDANCECHECK_MIN: float | None = 3.0

# =========================
# CANONICAL LABEL MAPS
# =========================

COND_TYPO_FIX = {
    "fasle_authoritative": "false_authoritative",
    "fasle-authoritative": "false_authoritative",
    "fasle authoritative": "false_authoritative",
    "false-authoritative": "false_authoritative",
    "false authoritative": "false_authoritative",
    "false_neutral": "false_neutral",
    "false-neutral": "false_neutral",
    "true_neutral": "true_neutral",
    "true-neutral": "true_neutral",
    "true_authoritative": "true_authoritative",
    "true-authoritative": "true_authoritative",
}

SHARE_MAP = {
    "Extremely unlikely": 0.00,
    "Unlikely": 0.25,
    "Neutral": 0.50,
    "Likely": 0.75,
    "Extremely likely": 1.00,
}
TRUTH_MAP = {
    "Not at all truthful": 1.0,
    "Slightly truthful": 2.0,
    "Moderately truthful": 3.0,
    "Very truthful": 4.0,
    "Completely truthful": 5.0,
}
VIG_MAP = {
    "Not at all vigilant": 1.0,
    "Slightly vigilant": 2.0,
    "Moderately vigilant": 3.0,
    "Very vigilant": 4.0,
    "Extremely vigilant": 5.0,
}
CONF_MAP = {
    "not confident at all": 1.0,
    "slightly confident": 2.0,
    "moderately confident": 3.0,
    "very confident": 4.0,
    "extremely confident": 5.0,
}
EPISTEMIC_MAP = {
    "Not at all carefully": 1.0,
    "Slightly carefully": 2.0,
    "Moderately carefully": 3.0,
    "Very carefully": 4.0,
    "Extremely carefully": 5.0,
}
AUTHORITY_MAP = {
    "Not at all authoritative": 1.0,
    "Slightly authoritative": 2.0,
    "Moderately authoritative": 3.0,
    "Very authoritative": 4.0,
    "Extremely authoritative": 5.0,
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

GENDER_MAP = {
    "Male": "Male",
    "Female": "Female",
    "Non-binary / third gender": "Non-binary / third gender",
    "Prefer not to say": "Prefer not to say",
    "Non-binary / third gender or Prefer not to say": "Non-binary / third gender",
}

EDU_NUM_MAP = {
    "High School": 1,
    "Bachelor": 2,
    "Master": 3,
    "PhD": 4,
    "Other": 5,
}

# =========================
# UTILITIES
# =========================

def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def norm_bool(x) -> bool:
    s = norm_str(x).upper()
    return s in {"TRUE", "1", "YES", "Y", "T"}

def to_float_mixed(series: pd.Series, label_map: Optional[Dict[str, float]] = None) -> pd.Series:
    def conv(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = norm_str(x)
        if label_map and s in label_map:
            return float(label_map[s])
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.apply(conv)

def map_strict(series: pd.Series, label_map: Dict[str, float], colname: str, noise: List[dict]) -> pd.Series:
    """
    Map string labels to floats. If unmapped non-missing values exist, record them in noise.
    Numeric-looking values are accepted as numeric.
    """
    out = np.empty(len(series), dtype="float64")
    out[:] = np.nan

    for idx, x in series.items():
        if pd.isna(x):
            continue
        s = norm_str(x)
        if s == "":
            continue
        if s in label_map:
            out[series.index.get_loc(idx)] = float(label_map[s])
            continue
        try:
            out[series.index.get_loc(idx)] = float(s)
        except ValueError:
            noise.append({"row_index": int(idx), "column": colname, "raw_value": s, "issue": "unmapped_label"})
            out[series.index.get_loc(idx)] = np.nan

    return pd.Series(out, index=series.index, dtype="float64")

def safe_mean_rowwise(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: np.nanmean(r.values.astype(float)), axis=1)

def normalize_indicator_value(v: str) -> str:
    s = v.lower().strip()
    s = s.replace(" ", "_").replace("-", "_")
    return COND_TYPO_FIX.get(s, s)

def infer_condition_from_indicators(df: pd.DataFrame, noise: List[dict]) -> pd.Series:
    """
    From 4 indicator columns, infer condition if exactly one indicator is present on a row.
    Returns a Series with values in {"True-Neutral","True-Authoritative","False-Authoritative","False-Neutral"} or "".
    """
    required = ["true_neutral", "true_authoritative", "false_authoritative", "false_neutral"]
    if not all(c in df.columns for c in required):
        return pd.Series([""] * len(df), index=df.index, dtype="string")

    tmp = df[required].copy()
    for c in required:
        tmp[c] = tmp[c].apply(norm_str)

    present_mask = tmp.ne("")

    count_present = present_mask.sum(axis=1)

    bucket_to_label = {
        "true_neutral": "True-Neutral",
        "true_authoritative": "True-Authoritative",
        "false_authoritative": "False-Authoritative",
        "false_neutral": "False-Neutral",
    }

    inferred = pd.Series([""] * len(df), index=df.index, dtype="string")

    # exactly one present -> choose that column name as the bucket
    one_mask = count_present == 1
    if one_mask.any():
        # idxmax works because booleans True/False; safe since exactly one True
        chosen_bucket = present_mask[one_mask].idxmax(axis=1)
        inferred.loc[one_mask] = chosen_bucket.map(bucket_to_label).astype("string")

    # ambiguous rows: >1 present
    amb_mask = count_present > 1
    for i in df.index[amb_mask]:
        noise.append({"row_index": int(i), "column": "condition", "raw_value": "multiple indicators set", "issue": "ambiguous_condition"})

    return inferred

# =========================
# MAIN CLEANING
# =========================

def clean():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH.resolve()}")

    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]

    noise: List[dict] = []

    # ---- Standardize the four condition indicator columns if they exist
    # Also handle misspelling column "fasle_authoritative"
    for c in ["true_neutral", "true_authoritative", "false_authoritative", "false_neutral", "fasle_authoritative"]:
        if c in df.columns:
            df[c] = df[c].apply(norm_str)

    if "fasle_authoritative" in df.columns:
        if "false_authoritative" not in df.columns:
            df["false_authoritative"] = ""
        moved = df["fasle_authoritative"].apply(norm_str)
        mask_move = (df["false_authoritative"].apply(norm_str) == "") & (moved != "")
        df.loc[mask_move, "false_authoritative"] = moved[mask_move]
        df = df.drop(columns=["fasle_authoritative"])

    # ---- Harmonize recaptcha column name variants
    rename_map = {}
    for c in list(df.columns):
        if c.strip().lower() == "q_recaptchascore":
            rename_map[c] = "q_recaptchascore"
    df = df.rename(columns=rename_map)

    # ---- Core fields: finished, recaptcha, attendancecheck
    if "finished" in df.columns:
        df["finished"] = df["finished"].apply(norm_bool)
    else:
        df["finished"] = True
        noise.append({"row_index": -1, "column": "finished", "raw_value": "", "issue": "missing_column_defaulted_true"})

    if "q_recaptchascore" in df.columns:
        df["q_recaptchascore"] = to_float_mixed(df["q_recaptchascore"], None)
    else:
        df["q_recaptchascore"] = np.nan
        noise.append({"row_index": -1, "column": "q_recaptchascore", "raw_value": "", "issue": "missing_column"})

    if "attendancecheck" in df.columns:
        df["attendancecheck"] = to_float_mixed(df["attendancecheck"], None)
    else:
        df["attendancecheck"] = np.nan
        noise.append({"row_index": -1, "column": "attendancecheck", "raw_value": "", "issue": "missing_column"})

    # ---- Condition: prefer explicit 'condition' column; fill missing from indicators
    if "condition" in df.columns:
        df["condition"] = df["condition"].apply(norm_str)
    else:
        df["condition"] = ""
        noise.append({"row_index": -1, "column": "condition", "raw_value": "", "issue": "missing_column"})

    inferred = infer_condition_from_indicators(df, noise)
    mask_empty_cond = df["condition"].apply(norm_str).eq("")
    df.loc[mask_empty_cond, "condition"] = inferred.loc[mask_empty_cond].fillna("").astype(str)

    # ---- is_true_condition (NumPy 2.x safe)
    cond_s = df["condition"].astype("string")
    df["is_true_condition"] = pd.Series(pd.NA, index=df.index, dtype="string")

    mask_true = cond_s.str.startswith("True", na=False)
    mask_false = cond_s.str.startswith("False", na=False)

    df.loc[mask_true, "is_true_condition"] = "True Condition"
    df.loc[mask_false, "is_true_condition"] = "False Condition"    
    

    # ---- context_authority: use existing, else infer from condition
    if "context_authority" in df.columns:
        df["context_authority"] = df["context_authority"].apply(norm_str)
    else:
        df["context_authority"] = ""

    ctx_empty = df["context_authority"].apply(norm_str).eq("")
    cond_txt = df["condition"].astype("string")
    df.loc[ctx_empty & cond_txt.str.contains("Authoritative", na=False), "context_authority"] = "Authoritative setting"
    df.loc[ctx_empty & cond_txt.str.contains("Neutral", na=False), "context_authority"] = "Neutral setting"

    # ---- correct_guess
    if "correct_guess" in df.columns:
        df["correct_guess"] = to_float_mixed(df["correct_guess"], None)
    else:
        df["correct_guess"] = np.nan

    # ---- Psychometric items
    trust_items = [c for c in ["trust_authority_3","trust_authority_4","trust_authority_5","trust_authority_6","trust_authority_7"] if c in df.columns]
    media_items = [c for c in ["media_literacy_1","media_literacy_2","media_literacy_3","media_literacy_4","media_literacy_5"] if c in df.columns]

    for c in trust_items:
        df[c] = map_strict(df[c], LIKERT5_AGREE, c, noise)
    for c in media_items:
        df[c] = map_strict(df[c], LIKERT5_AGREE, c, noise)

    if trust_items:
        df["trust_authority"] = safe_mean_rowwise(df[trust_items])
    else:
        df["trust_authority"] = np.nan
        noise.append({"row_index": -1, "column": "trust_authority", "raw_value": "", "issue": "missing_items"})

    if media_items:
        df["media_literacy"] = safe_mean_rowwise(df[media_items])
    else:
        df["media_literacy"] = np.nan
        noise.append({"row_index": -1, "column": "media_literacy", "raw_value": "", "issue": "missing_items"})

    if "env_trust" not in df.columns:
        df["env_trust"] = df["trust_authority"]
    if "env_media" not in df.columns:
        df["env_media"] = df["media_literacy"]

    # ---- Familiarity
    if "familiarity_academic" in df.columns:
        df["familiarity_academic"] = df["familiarity_academic"].apply(norm_str)
        df["fam_academic_score"] = map_strict(df["familiarity_academic"], FAM_MAP, "familiarity_academic", noise)
    else:
        df["familiarity_academic"] = np.nan
        df["fam_academic_score"] = np.nan

    # ---- Outcomes
    if "share_willingness" in df.columns and "share_score" not in df.columns:
        df["share_willingness"] = df["share_willingness"].apply(norm_str)
        df["share_score"] = map_strict(df["share_willingness"], SHARE_MAP, "share_willingness", noise)
    elif "share_score" in df.columns:
        df["share_score"] = map_strict(df["share_score"], SHARE_MAP, "share_score", noise)
        if "share_willingness" not in df.columns:
            df["share_willingness"] = np.nan
    else:
        df["share_willingness"] = np.nan
        df["share_score"] = np.nan

    if "perceived_truth" in df.columns and "truth_score" not in df.columns:
        df["perceived_truth"] = df["perceived_truth"].apply(norm_str)
        df["truth_score"] = map_strict(df["perceived_truth"], TRUTH_MAP, "perceived_truth", noise)
    elif "truth_score" in df.columns:
        df["truth_score"] = map_strict(df["truth_score"], TRUTH_MAP, "truth_score", noise)
        if "perceived_truth" not in df.columns:
            df["perceived_truth"] = np.nan
    else:
        df["perceived_truth"] = np.nan
        df["truth_score"] = np.nan

    # epistemic_vigilance score
    if "epistemic_vigilance" in df.columns:
        df["epistemic_vigilance"] = df["epistemic_vigilance"].apply(norm_str)
        df["epistemic_vigilance_score"] = map_strict(df["epistemic_vigilance"], EPISTEMIC_MAP, "epistemic_vigilance", noise)
    else:
        df["epistemic_vigilance"] = np.nan
        df["epistemic_vigilance_score"] = np.nan

    # vigilance_score numeric: map if label exists; otherwise use epistemic_vigilance_score as fallback
    if "vigilance_score" in df.columns:
        df["vigilance_score"] = df["vigilance_score"].apply(norm_str)
        df["vigilance_score_num"] = map_strict(df["vigilance_score"], VIG_MAP, "vigilance_score", noise)
    else:
        df["vigilance_score"] = np.nan
        df["vigilance_score_num"] = df["epistemic_vigilance_score"]

    # ---- Manipulation / authority checks
    if "manipulationcheck" in df.columns:
        df["manipulationcheck"] = df["manipulationcheck"].apply(norm_str)
        df["authority_score"] = map_strict(df["manipulationcheck"], AUTHORITY_MAP, "manipulationcheck", noise)
    else:
        df["manipulationcheck"] = np.nan
        df["authority_score"] = np.nan

    # ---- Confidence
    if "confidence_judgment" in df.columns:
        df["confidence_judgment"] = df["confidence_judgment"].apply(norm_str)
        df["confidence_score"] = map_strict(df["confidence_judgment"], CONF_MAP, "confidence_judgment", noise)
    else:
        df["confidence_judgment"] = np.nan
        df["confidence_score"] = np.nan

    # ---- Student
    if "student" in df.columns:
        df["student"] = df["student"].apply(norm_str)
        df["student_num"] = df["student"].map(lambda x: 1 if x == "Yes" else (0 if x == "No" else np.nan))
        bad = ~df["student"].isin(["Yes", "No", ""])
        for i in df.index[bad]:
            noise.append({"row_index": int(i), "column": "student", "raw_value": df.at[i, "student"], "issue": "unexpected_student_value"})
    else:
        df["student"] = np.nan
        df["student_num"] = np.nan

    # ---- Gender
    if "gender_cat" in df.columns:
        gsrc = df["gender_cat"].apply(norm_str)
    elif "q35" in df.columns:
        gsrc = df["q35"].apply(norm_str)
    else:
        gsrc = pd.Series([""] * len(df), index=df.index)

    def norm_gender(x: str):
        s = norm_str(x)
        if s in GENDER_MAP:
            return GENDER_MAP[s]
        if "Non-binary" in s and "Prefer not" in s:
            return "Non-binary / third gender"
        if s != "":
            # record noise with real row id
            return np.nan
        return np.nan

    df["gender_cat"] = gsrc.apply(norm_gender)
    # log unmapped genders
    for i, v in gsrc.items():
        sv = norm_str(v)
        if sv != "" and sv not in GENDER_MAP and not ("Non-binary" in sv and "Prefer not" in sv):
            noise.append({"row_index": int(i), "column": "gender_cat", "raw_value": sv, "issue": "unmapped_gender"})

    # ---- Age
    if "agecat" in df.columns:
        df["agecat"] = df["agecat"].apply(norm_str)
    else:
        df["agecat"] = ""

    for c in ["age_1","age_2","age_3","age_4","age_5"]:
        if c not in df.columns:
            df[c] = 0

    # ---- Education numeric
    if "edu_level_num" in df.columns:
        df["edu_level_num"] = to_float_mixed(df["edu_level_num"], None)
    else:
        # best-effort (only works if you have a column named 'edu_level' with Bachelor/Master/PhD/etc.)
        if "edu_level" in df.columns:
            src = df["edu_level"].apply(norm_str)
        else:
            src = pd.Series([""] * len(df), index=df.index)
        df["edu_level_num"] = src.map(lambda x: EDU_NUM_MAP.get(x, np.nan))
        bad = (src != "") & (~src.isin(list(EDU_NUM_MAP.keys())))
        for i in df.index[bad]:
            noise.append({"row_index": int(i), "column": "edu_level_num", "raw_value": src.at[i], "issue": "unmapped_edu_level"})

    # ---- Attend check convenience
    if "attend_check" not in df.columns:
        df["attend_check"] = df["attendancecheck"]

    # =========================
    # APPLY QUALITY FILTERS
    # =========================
    before = len(df)

    if FILTER_REQUIRE_FINISHED:
        df = df[df["finished"] == True]
    if FILTER_RECAPTCHA_MIN is not None:
        df = df[df["q_recaptchascore"].isna() | (df["q_recaptchascore"] >= FILTER_RECAPTCHA_MIN)]
    if FILTER_ATTENDANCECHECK_MIN is not None:
        df = df[df["attendancecheck"].isna() | (df["attendancecheck"] >= FILTER_ATTENDANCECHECK_MIN)]

    print(f"[INFO] Rows after filters: {before} -> {len(df)}")

    # =========================
    # OUTPUT COLUMNS (requested order)
    # =========================
    out_cols = [
        "finished","q_recaptchascore","true_neutral","true_authoritative","false_authoritative","false_neutral",
        "attendancecheck","guess_question","perceived_truth","share_willingness","confidence_judgment","epistemic_vigilance",
        "manipulationcheck","trust_authority_3","trust_authority_4","trust_authority_5","trust_authority_6","trust_authority_7",
        "media_literacy_1","media_literacy_2","media_literacy_3","media_literacy_4","media_literacy_5",
        "q34","q35","q36","country","familiarity_academic","student","condition","attend_check",
        "authority_score","context_authority","share_score","truth_score","vigilance_score",
        "actual_true_statements","correct_guess","trust_authority","media_literacy","env_trust","env_media",
        "agecat","age_1","age_2","age_3","age_4","age_5","fam_academic_score","continent","student_num",
        "gender_cat","confidence_score","is_true_condition","edu_level_num"
    ]

    # Ensure missing columns exist so output is stable
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Always write noise report if noise exists
    noise_df = pd.DataFrame(noise)
    if len(noise_df) > 0:
        OUT_NOISE.parent.mkdir(parents=True, exist_ok=True)
        noise_df.to_csv(OUT_NOISE, index=False)
        print(f"[INFO] Wrote noise report: {OUT_NOISE.resolve()} ({len(noise_df)} issues)")

        if REQUIRE_NOISE_APPROVAL and not APPROVE_NOISE:
            raise SystemExit(
                "[STOP] Noise detected. Review noise_report.csv, then set APPROVE_NOISE=True and rerun."
            )

    # Write clean
    df_out = df[out_cols].copy()
    OUT_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CLEAN, index=False)
    print(f"[INFO] Wrote clean CSV: {OUT_CLEAN.resolve()}")

    if len(noise_df) == 0:
        print("[INFO] No noise detected.")

if __name__ == "__main__":
    clean()

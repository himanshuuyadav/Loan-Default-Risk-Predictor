from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features described in the project plan."""

    frame = df.copy()

    if "annual_inc" in frame.columns:
        annual_income = frame["annual_inc"].clip(lower=0)
        frame["annual_inc_log"] = np.log1p(annual_income)

    if "loan_amnt" in frame.columns:
        loan_amount = frame["loan_amnt"].clip(lower=0)
        frame["loan_amnt_log"] = np.log1p(loan_amount)

    if {"fico_range_low", "fico_range_high"}.issubset(frame.columns):
        frame["fico_avg"] = (
            frame["fico_range_low"].fillna(0) + frame["fico_range_high"].fillna(0)
        ) / 2
        frame = frame.drop(columns=["fico_range_low", "fico_range_high"])

    annual_income_safe = frame.get("annual_inc", pd.Series(1.0, index=frame.index)).replace(0, np.nan)

    if {"loan_amnt", "annual_inc"}.issubset(frame.columns):
        frame["loan_to_income"] = frame["loan_amnt"] / annual_income_safe

    if {"installment", "annual_inc"}.issubset(frame.columns):
        frame["installment_to_income"] = frame["installment"] / annual_income_safe

    if "dti" in frame.columns:
        frame["dti_band"] = pd.cut(
            frame["dti"].clip(lower=0),
            bins=[0, 10, 20, 30, 40, np.inf],
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
            include_lowest=True,
        )

    if "pub_rec" in frame.columns or "delinq_2yrs" in frame.columns:
        pub_rec = frame.get("pub_rec", pd.Series(0, index=frame.index)).fillna(0)
        delinq = frame.get("delinq_2yrs", pd.Series(0, index=frame.index)).fillna(0)
        frame["has_derog_record"] = ((pub_rec > 0) | (delinq > 0)).astype(int)

    if "revol_util" in frame.columns:
        frame["high_revol_util"] = (frame["revol_util"].fillna(0) > 75).astype(int)

    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)

    return frame

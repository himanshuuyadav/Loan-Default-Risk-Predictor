from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.features import engineer_features

VALID_LOAN_STATUSES = ["Fully Paid", "Charged Off", "Default"]
SELECTED_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "emp_length",
    "home_ownership",
    "purpose",
    "grade",
    "open_acc",
    "revol_util",
    "pub_rec",
    "delinq_2yrs",
    "mort_acc",
    "installment",
    "total_acc",
]
NOMINAL_COLUMNS = ["home_ownership", "purpose", "dti_band"]
DTI_BAND_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"]


def load_data(filepath: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the raw Lending Club dataset."""

    return pd.read_csv(filepath, low_memory=False, nrows=nrows)


def filter_valid_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only loans with completed outcomes."""

    return df[df["loan_status"].isin(VALID_LOAN_STATUSES)].copy()


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map the Lending Club loan statuses into a binary default flag."""

    frame = df.copy()
    frame["default"] = frame["loan_status"].map(
        {"Fully Paid": 0, "Charged Off": 1, "Default": 1}
    )
    return frame


def prepare_raw_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select the project feature set plus the target."""

    required = SELECTED_FEATURES + ["default"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df[required].copy()


def build_model_frame(filepath: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the raw CSV and prepare the filtered training frame."""

    df = load_data(filepath, nrows=nrows)
    df = filter_valid_loans(df)
    df = create_binary_target(df)
    return prepare_raw_model_frame(df)


@dataclass
class LoanDefaultPreprocessor:
    """Reusable preprocessing pipeline for training and inference."""

    feature_columns_: List[str] = field(default_factory=list)
    numeric_fill_values_: Dict[str, float] = field(default_factory=dict)
    categorical_fill_values_: Dict[str, str] = field(default_factory=dict)
    drop_threshold_: int = 0

    def fit(self, df: pd.DataFrame) -> "LoanDefaultPreprocessor":
        frame = self._ensure_target_frame(df)
        cleaned = self._drop_sparse_rows(frame, fit=True)
        cleaned = self._clean_columns(cleaned)

        numeric_columns = [
            column
            for column in cleaned.select_dtypes(include=[np.number]).columns
            if column != "default"
        ]
        categorical_columns = cleaned.select_dtypes(exclude=[np.number]).columns.tolist()

        self.numeric_fill_values_ = {
            column: float(cleaned[column].median()) for column in numeric_columns
        }
        self.categorical_fill_values_ = {
            column: str(cleaned[column].mode(dropna=True).iloc[0])
            for column in categorical_columns
            if not cleaned[column].mode(dropna=True).empty
        }

        transformed = self._transform_core(cleaned, fit=True)
        self.feature_columns_ = [column for column in transformed.columns if column != "default"]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns_:
            raise ValueError("Preprocessor must be fitted before calling transform().")

        frame = self._ensure_target_frame(df)
        transformed = self._transform_core(frame, fit=False)
        feature_frame = transformed.drop(columns=["default"], errors="ignore")
        feature_frame = feature_frame.reindex(columns=self.feature_columns_, fill_value=0.0)

        if "default" in transformed.columns:
            feature_frame["default"] = transformed["default"].values

        return feature_frame

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def transform_applicant(self, applicant_data: pd.DataFrame | Dict) -> pd.DataFrame:
        """Transform raw applicant data into model-ready features."""

        if isinstance(applicant_data, dict):
            frame = pd.DataFrame([applicant_data])
        else:
            frame = applicant_data.copy()
        transformed = self.transform(frame)
        return transformed.reindex(columns=self.feature_columns_, fill_value=0.0)

    def _ensure_target_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        for column in SELECTED_FEATURES:
            if column not in frame.columns:
                frame[column] = np.nan
        ordered_columns = SELECTED_FEATURES + (["default"] if "default" in frame.columns else [])
        return frame[ordered_columns].copy()

    def _drop_sparse_rows(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        frame = df.copy()
        if fit:
            self.drop_threshold_ = int(np.ceil(len(frame.columns) * 0.5))
        if self.drop_threshold_:
            frame = frame.dropna(thresh=self.drop_threshold_)
        return frame

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()

        for percent_column in ["int_rate", "revol_util"]:
            if percent_column in frame.columns:
                frame[percent_column] = (
                    frame[percent_column]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .replace({"nan": np.nan, "None": np.nan})
                )
                frame[percent_column] = pd.to_numeric(frame[percent_column], errors="coerce")

        if "emp_length" in frame.columns:
            frame["emp_length"] = (
                frame["emp_length"]
                .astype(str)
                .str.extract(r"(\d+)")
                .iloc[:, 0]
            )
            frame["emp_length"] = pd.to_numeric(frame["emp_length"], errors="coerce")

        numeric_columns = [
            column
            for column in frame.columns
            if column != "default" and pd.api.types.is_numeric_dtype(frame[column])
        ]
        categorical_columns = [
            column
            for column in frame.columns
            if column not in numeric_columns and column != "default"
        ]

        for column in numeric_columns:
            fill_value = self.numeric_fill_values_.get(column)
            if fill_value is None:
                fill_value = float(frame[column].median()) if not frame[column].dropna().empty else 0.0
            frame[column] = frame[column].fillna(fill_value)

        for column in categorical_columns:
            fill_value = self.categorical_fill_values_.get(column)
            if fill_value is None:
                mode = frame[column].mode(dropna=True)
                fill_value = str(mode.iloc[0]) if not mode.empty else "Unknown"
            frame[column] = frame[column].fillna(fill_value)

        return frame

    def _transform_core(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        frame = self._drop_sparse_rows(df, fit=fit)
        frame = self._clean_columns(frame)

        if "grade" in frame.columns:
            grade_map = {grade: index for index, grade in enumerate(["A", "B", "C", "D", "E", "F", "G"])}
            frame["grade_encoded"] = frame["grade"].map(grade_map).fillna(-1).astype(float)
            frame = frame.drop(columns=["grade"])

        frame = engineer_features(frame)

        categorical_to_encode = [column for column in NOMINAL_COLUMNS if column in frame.columns]
        frame = pd.get_dummies(frame, columns=categorical_to_encode, drop_first=True, dtype=float)

        numeric_columns = [
            column
            for column in frame.columns
            if column != "default" and pd.api.types.is_numeric_dtype(frame[column])
        ]
        frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)

        for column in numeric_columns:
            fallback = self.numeric_fill_values_.get(column, 0.0)
            if pd.isna(fallback):
                fallback = 0.0
            frame[column] = frame[column].fillna(fallback)

        return frame

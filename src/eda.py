from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing import build_model_frame


def run_eda(data_path: str | Path, reports_dir: str | Path, nrows: int | None = None):
    """Generate the main EDA figures and summary stats from the plan."""

    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = build_model_frame(data_path, nrows=nrows)

    plt.figure(figsize=(6, 4))
    df["default"].value_counts().sort_index().plot(kind="bar", color=["steelblue", "tomato"])
    plt.title("Loan Default Distribution")
    plt.xlabel("Default (0=No, 1=Yes)")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png")
    plt.close()

    numeric_cols = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low"]
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(20, 4))
    for index, column in enumerate(numeric_cols):
        cleaned = (
            df[column]
            .astype(str)
            .str.replace("%", "", regex=False)
            .replace({"nan": pd.NA, "None": pd.NA})
        )
        cleaned = pd.to_numeric(cleaned, errors="coerce")
        cleaned.plot(kind="hist", bins=50, ax=axes[index], color="steelblue", edgecolor="white")
        axes[index].set_title(column)
    fig.tight_layout()
    fig.savefig(figures_dir / "numeric_distributions.png")
    plt.close(fig)

    grade_default = df.groupby("grade", dropna=False)["default"].mean().sort_index()
    plt.figure(figsize=(8, 4))
    grade_default.plot(kind="bar", color="salmon")
    plt.title("Default Rate by Loan Grade")
    plt.ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(figures_dir / "default_by_grade.png")
    plt.close()

    corr_cols = [
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "fico_range_low",
        "open_acc",
        "revol_util",
        "default",
    ]
    correlation_frame = df[corr_cols].copy()
    for column in ["int_rate", "revol_util"]:
        correlation_frame[column] = pd.to_numeric(
            correlation_frame[column].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_frame.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png")
    plt.close()

    missing = df.isnull().mean().sort_values(ascending=False)
    missing_significant = missing[missing > 0.05]
    if not missing_significant.empty:
        plt.figure(figsize=(10, 6))
        missing_significant.sort_values().plot(kind="barh", color="orange")
        plt.title("Features with >5% Missing Values")
        plt.xlabel("Missing Fraction")
        plt.tight_layout()
        plt.savefig(figures_dir / "missing_values.png")
        plt.close()

    summary = {
        "shape": list(df.shape),
        "default_rate": float(df["default"].mean()),
        "default_counts": df["default"].value_counts().sort_index().to_dict(),
        "missing_fraction": missing.to_dict(),
        "grade_default_rate": grade_default.to_dict(),
    }

    with (reports_dir / "eda_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run exploratory data analysis for the loan default dataset.")
    parser.add_argument("--data-path", required=True, help="Path to the raw Lending Club CSV file.")
    parser.add_argument("--reports-dir", default="reports", help="Directory for saving EDA outputs.")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick exploration.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda(args.data_path, args.reports_dir, nrows=args.nrows)

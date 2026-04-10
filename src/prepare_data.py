from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing import LoanDefaultPreprocessor, build_model_frame


def prepare_dataset(
    data_path: str | Path,
    output_path: str | Path = "data/processed/prepared_dataset.csv",
    nrows: int | None = None,
):
    """Build the cleaned and engineered dataset before training."""

    frame = build_model_frame(data_path, nrows=nrows)
    preprocessor = LoanDefaultPreprocessor()
    processed = preprocessor.fit_transform(frame)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)

    return {
        "rows": int(processed.shape[0]),
        "columns": int(processed.shape[1]),
        "output_path": str(output_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the cleaned and engineered modeling dataset.")
    parser.add_argument("--data-path", required=True, help="Path to the raw Lending Club CSV file.")
    parser.add_argument(
        "--output-path",
        default="data/processed/prepared_dataset.csv",
        help="Where to write the prepared dataset CSV.",
    )
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick experiments.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = prepare_dataset(args.data_path, output_path=args.output_path, nrows=args.nrows)
    print(summary)

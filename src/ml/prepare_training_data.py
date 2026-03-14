import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = [
    "track_genre",
    "popularity",
    "danceability",
    "energy",
    "valence",
    "tempo",
]


def _validate_source_columns(data: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"Input file is missing required columns for feature prep: {missing_columns}"
        )


def _build_feature_frame(data: pd.DataFrame, min_genre_frequency: int) -> pd.DataFrame:
    prepared = data[REQUIRED_COLUMNS].copy()
    prepared["track_genre"] = (
        prepared["track_genre"].astype(str).str.strip().str.lower()
    )
    prepared["popularity"] = prepared["popularity"].astype(float)

    # Group rare genres to reduce train/inference mismatch on unseen labels.
    genre_counts = prepared["track_genre"].value_counts()
    frequent_genres = set(
        genre_counts[genre_counts >= min_genre_frequency].index.tolist()
    )
    prepared["track_genre"] = prepared["track_genre"].where(
        prepared["track_genre"].isin(frequent_genres), "__other__"
    )

    tempo = prepared["tempo"].astype(float)
    prepared["tempo_log"] = np.log1p(tempo)
    prepared["energy_x_danceability"] = prepared["energy"].astype(float) * prepared[
        "danceability"
    ].astype(float)
    prepared["mood_index"] = 0.6 * prepared["valence"].astype(float) + 0.4 * prepared[
        "energy"
    ].astype(float)
    return prepared


def run(
    input_path: Path,
    train_output_path: Path,
    valid_output_path: Path,
    report_path: Path,
    *,
    test_size: float,
    random_state: int,
    min_genre_frequency: int,
) -> None:
    source = pd.read_csv(input_path)
    _validate_source_columns(source)
    prepared = _build_feature_frame(source, min_genre_frequency=min_genre_frequency)

    train_df, valid_df = train_test_split(
        prepared, test_size=test_size, random_state=random_state, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_output_path, index=False)
    valid_df.to_csv(valid_output_path, index=False)

    report = {
        "source_rows": int(len(source)),
        "prepared_rows": int(len(prepared)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "feature_columns": [
            "track_genre",
            "danceability",
            "energy",
            "valence",
            "tempo",
            "tempo_log",
            "energy_x_danceability",
            "mood_index",
        ],
        "target_column": "popularity",
        "test_size": test_size,
        "random_state": random_state,
        "min_genre_frequency": min_genre_frequency,
        "retained_genres": int(train_df["track_genre"].nunique()),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/validation datasets for popularity modeling."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to clean CSV")
    parser.add_argument(
        "--train-output", type=Path, required=True, help="Path to train CSV"
    )
    parser.add_argument(
        "--valid-output", type=Path, required=True, help="Path to validation CSV"
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to JSON report with feature engineering statistics",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        required=True,
        help="Validation share in range (0, 1).",
    )
    parser.add_argument(
        "--random-state", type=int, required=True, help="Random seed for data split."
    )
    parser.add_argument(
        "--min-genre-frequency",
        type=int,
        required=True,
        help="Minimum per-genre rows to keep a dedicated category.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        input_path=args.input,
        train_output_path=args.train_output,
        valid_output_path=args.valid_output,
        report_path=args.report,
        test_size=args.test_size,
        random_state=args.random_state,
        min_genre_frequency=args.min_genre_frequency,
    )


if __name__ == "__main__":
    main()

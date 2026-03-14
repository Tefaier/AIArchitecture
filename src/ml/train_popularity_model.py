import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "popularity"


def _build_model(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def _collect_feature_importances(model: Pipeline, *, top_k: int = 12) -> list[dict]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    regressor: RandomForestRegressor = model.named_steps["regressor"]
    feature_names = preprocessor.get_feature_names_out().tolist()
    importances = regressor.feature_importances_.tolist()

    ranked = sorted(
        (
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(feature_names, importances, strict=True)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranked[:top_k]


def run(
    train_path: Path,
    valid_path: Path,
    model_path: Path,
    metrics_path: Path,
    predictions_path: Path,
    *,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
) -> None:
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in valid_df.columns:
        raise ValueError(f"Target column `{TARGET_COLUMN}` is required in train/valid.")

    feature_columns = [col for col in train_df.columns if col != TARGET_COLUMN]
    numeric_features = [
        col for col in feature_columns if pd.api.types.is_numeric_dtype(train_df[col])
    ]
    categorical_features = [
        col for col in feature_columns if col not in numeric_features
    ]

    model = _build_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(float)
    x_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN].astype(float)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)

    baseline_value = float(y_train.mean())
    baseline_pred = np.full_like(y_valid.to_numpy(), fill_value=baseline_value)

    rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
    mae = float(mean_absolute_error(y_valid, y_pred))
    r2 = float(r2_score(y_valid, y_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_valid, baseline_pred)))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    predictions = pd.DataFrame({"y_true": y_valid, "y_pred": y_pred})
    predictions["abs_error"] = (predictions["y_true"] - predictions["y_pred"]).abs()
    predictions.to_csv(predictions_path, index=False)

    metrics = {
        "rows_train": int(len(train_df)),
        "rows_valid": int(len(valid_df)),
        "target_column": TARGET_COLUMN,
        "model_name": "RandomForestRegressor",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "baseline_mean_popularity": baseline_value,
        "baseline_rmse": baseline_rmse,
        "rmse_improvement_vs_baseline": baseline_rmse - rmse,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "top_feature_importances": _collect_feature_importances(model),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a scikit-learn popularity predictor on engineered features."
    )
    parser.add_argument("--train", type=Path, required=True, help="Path to train CSV")
    parser.add_argument(
        "--valid", type=Path, required=True, help="Path to validation CSV"
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to output serialized model"
    )
    parser.add_argument(
        "--metrics", type=Path, required=True, help="Path to JSON metrics file"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to CSV with validation predictions",
    )
    parser.add_argument(
        "--random-state", type=int, required=True, help="Random seed for estimator."
    )
    parser.add_argument(
        "--n-estimators", type=int, required=True, help="Number of trees in forest."
    )
    parser.add_argument(
        "--max-depth", type=int, required=True, help="Maximum tree depth."
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        required=True,
        help="Minimum number of samples per leaf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        train_path=args.train,
        valid_path=args.valid,
        model_path=args.model,
        metrics_path=args.metrics,
        predictions_path=args.predictions,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )


if __name__ == "__main__":
    main()

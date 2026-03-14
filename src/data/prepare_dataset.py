import argparse
import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

RAW_COLUMNS = [
    "track_id",
    "artists",
    "track_name",
    "track_genre",
    "popularity",
    "danceability",
    "energy",
    "valence",
    "tempo",
]


def build_schema() -> pa.DataFrameSchema:
    """Создает Pandera-схему для очищенного Spotify-датасета.

    Returns:
        pa.DataFrameSchema: Схема с ограничениями по типам и диапазонам.
    """
    return pa.DataFrameSchema(
        {
            "track_id": pa.Column(str, nullable=False),
            "artists": pa.Column(str, nullable=False),
            "track_name": pa.Column(str, nullable=False),
            "track_genre": pa.Column(str, nullable=False),
            "popularity": pa.Column(
                int, checks=pa.Check.in_range(0, 100), nullable=False
            ),
            "danceability": pa.Column(
                float, checks=pa.Check.in_range(0.0, 1.0), nullable=False
            ),
            "energy": pa.Column(
                float, checks=pa.Check.in_range(0.0, 1.0), nullable=False
            ),
            "valence": pa.Column(
                float, checks=pa.Check.in_range(0.0, 1.0), nullable=False
            ),
            "tempo": pa.Column(
                float, checks=pa.Check.in_range(1.0, 400.0), nullable=False
            ),
        },
        strict=True,
        coerce=True,
    )


def validate_and_filter_invalid_rows(
    schema: pa.DataFrameSchema, data: pd.DataFrame, drop_invalid_rows: bool
) -> tuple[pd.DataFrame, int, list[dict]]:
    """Проверяет датафрейм по схеме и при необходимости удаляет невалидные строки.

    Args:
        schema: Pandera-схема для валидации.
        data: Таблица для проверки.
        drop_invalid_rows: Флаг мягкой валидации. Если `True`, невалидные строки
            исключаются из результата. Если `False`, исключение пробрасывается наружу.

    Returns:
        tuple[pd.DataFrame, int, list[dict]]: Валидный датафрейм, количество удаленных
        строк и сводка по проваленным проверкам.

    Raises:
        SchemaErrors: Если валидация не пройдена и включен строгий режим, либо если
        невозможно безопасно определить индексы невалидных строк.
    """
    try:
        validated = schema.validate(data, lazy=True)
        return validated, 0, []
    except SchemaErrors as exc:
        if not drop_invalid_rows:
            raise

        failure_cases = exc.failure_cases.copy()
        if "index" not in failure_cases.columns:
            raise

        invalid_indexes = set(failure_cases["index"].dropna().tolist())
        if not invalid_indexes:
            raise

        filtered = data.loc[~data.index.isin(invalid_indexes)].reset_index(drop=True)
        validated = schema.validate(filtered, lazy=True)

        summary = (
            failure_cases.dropna(subset=["index"])
            .assign(
                column=lambda df: df["column"].astype(str),
                check=lambda df: df["check"].astype(str),
            )
            .groupby(["column", "check"], dropna=False)["index"]
            .nunique()
            .reset_index(name="failed_rows")
            .sort_values("failed_rows", ascending=False)
        )
        summary_records = summary.to_dict(orient="records")
        return validated, len(invalid_indexes), summary_records


def run(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    drop_invalid_rows: bool,
) -> None:
    """Запускает полный процесс очистки и валидации входного датасета.

    Args:
        input_path: Путь к исходному CSV-файлу.
        output_path: Путь для сохранения очищенного CSV-файла.
        report_path: Путь для сохранения JSON-отчета с метриками очистки.
        drop_invalid_rows: Флаг мягкой валидации. Если `True`, невалидные строки
            отбрасываются.

    Raises:
        ValueError: Если во входном файле отсутствуют обязательные колонки.
        SchemaErrors: Если данные не проходят валидацию в строгом режиме.
    """
    source = pd.read_csv(input_path)

    missing_columns = sorted(set(RAW_COLUMNS) - set(source.columns))
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")

    data = source[RAW_COLUMNS].copy()
    total_rows = len(data)

    data = data.dropna(subset=RAW_COLUMNS)
    after_dropna_rows = len(data)

    data = data.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    after_dedup_rows = len(data)

    schema = build_schema()
    validated, removed_by_validation, validation_failures = (
        validate_and_filter_invalid_rows(
            schema, data, drop_invalid_rows=drop_invalid_rows
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output_path, index=False)

    report = {
        "source_rows": total_rows,
        "rows_after_dropna": after_dropna_rows,
        "rows_after_deduplicate": after_dedup_rows,
        "rows_after_validation": len(validated),
        "removed_rows_with_nulls": total_rows - after_dropna_rows,
        "removed_duplicate_rows": after_dropna_rows - after_dedup_rows,
        "removed_rows_by_validation": removed_by_validation,
        "drop_invalid_rows": drop_invalid_rows,
        "validation_failures": validation_failures,
        "output_path": str(output_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    """Читает параметры командной строки для этапа подготовки данных.

    Returns:
        argparse.Namespace: Распарсенные аргументы запуска.
    """
    parser = argparse.ArgumentParser(
        description="Validate and clean Spotify tracks dataset."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to raw CSV file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to clean CSV file"
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to JSON report with cleaning statistics",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid rows instead of dropping them after lazy validation.",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа CLI-скрипта подготовки датасета."""
    args = parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        drop_invalid_rows=not args.strict,
    )


if __name__ == "__main__":
    main()

"""Общие константы и пути проекта."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "popylarity_model.joblib"

__all__ = ["DEFAULT_MODEL_PATH", "PROJECT_ROOT"]

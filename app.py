from __future__ import annotations

import os

from src import DEFAULT_MODEL_PATH


def main() -> None:
    """Запускает HTTP API сервиса предсказания популярности из корня проекта."""
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Не найден модуль uvicorn. Установите зависимости проекта и "
            "запускайте из активированного виртуального окружения."
        ) from exc

    os.environ.setdefault("POPULARITY_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    uvicorn.run("src.service:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

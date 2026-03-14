# Семинар: DVC + Pandera + sklearn + HTTP API (Spotify)

Источник датасета: <https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download>

Образовательные результаты:

1. Умеют версионировать данные через DVC (`dvc add`, `dvc commit`, `dvc push`).
2. Умеют подключать удаленное S3-хранилище к DVC (`dvc remote add`, `dvc remote modify`).
3. Умеют воспроизводить end-to-end пайплайн (подготовка данных + обучение модели).
4. Понимают, зачем нужно версионирование данных в ML-проекте.

Проект включает:
1. Пайплайн подготовки данных на Python + Pandera.
2. Пайплайн фичеинжиниринга и обучения модели популярности на `scikit-learn`.
3. HTTP API сервис предсказания популярности треков по фичам.

## Структура проекта

```text
.
├── datasets/dataset.csv.dvc          # raw датасет под DVC
├── dvc.yaml                          # multi-stage пайплайн DVC
├── params.yaml                       # гиперпараметры split/feature/train
├── src/data/prepare_dataset.py       # очистка и валидация Pandera
├── src/ml/prepare_training_data.py   # фичеинжиниринг + train/valid split
├── src/ml/train_popularity_model.py  # обучение sklearn модели
└── src/service/app.py                # FastAPI сервис
```

## 1. Подготовка окружения (Linux/macOS)

Требования:
1. `python == 3.12`
2. `uv`
3. `git`
4. `Секреты S3 -> .env`

Установка зависимостей:

```bash
uv sync
```

Если lock-файл устарел:

```bash
uv lock
uv sync
```

Установка pre-commit хуков:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 2. Подготовка raw данных

Ожидается, что файл уже скачан и лежит в `datasets/dataset.csv`.

Проверка:

```bash
ls -lh datasets/dataset.csv
```

Добавление raw данных в DVC:

```bash
uv run dvc add datasets/dataset.csv
git add datasets/dataset.csv.dvc .gitignore
git commit -m "data: add raw spotify dataset"
```

## 3. Настройка S3 remote для DVC

1. Скопируйте переменные окружения:

```bash
cp .env.example .env
```

2. Заполните в `.env` значения:
1. `AWS_ACCESS_KEY_ID`
2. `AWS_SECRET_ACCESS_KEY`
3. `AWS_ENDPOINT_URL`
4. `AWS_BUCKET_NAME`
5. `AWS_TARGET_DIR`
6. `AWS_REGION`

3. Подключите remote:

```bash
uv run dvc remote add -d storage s3://$AWS_BUCKET_NAME/$AWS_TARGET_DIR
uv run dvc remote modify storage endpointurl $AWS_ENDPOINT_URL
uv run dvc remote modify storage region $AWS_REGION
```

Проверка:

```bash
uv run dvc remote list
```

## 4. Воспроизведение ML-пайплайна (`dvc repro`)

В проекте есть три стадии:
1. Читает `datasets/dataset.csv`.
2. Удаляет строки с `null` в ключевых колонках.
3. Удаляет дубликаты по `track_id`.
4. Делает lazy-валидацию через Pandera.
5. Невалидные строки (например, `tempo=0`) автоматически отбрасываются.
6. Сохраняет `data/processed/clean_tracks.csv` и `reports/data_prep_report.json`.
7. Готовит train/valid с engineered-фичами:
   1. `tempo_log`
   2. `energy_x_danceability`
   3. `mood_index`
8. Обучает `RandomForestRegressor` для предсказания `popularity`.
9. Сохраняет `models/popylarity_model.joblib`, `reports/model_metrics.json` и предсказания валидации.

Запуск:

```bash
uv run dvc repro
```

Проверка результатов:

```bash
uv run dvc status
cat reports/data_prep_report.json
cat reports/feature_report.json
cat reports/model_metrics.json
```

Если нужно падать при первой невалидной записи (без авто-очистки), используйте строгий режим:

```bash
uv run python -m src.data.prepare_dataset \
  --input datasets/dataset.csv \
  --output data/processed/clean_tracks.csv \
  --report reports/data_prep_report.json \
  --strict
```

Зафиксировать изменения пайплайна:

```bash
git add dvc.yaml dvc.lock params.yaml
git commit -m "pipeline: add feature engineering and sklearn training"
```

## 5. Параметры экспериментов (`params.yaml`)

Параметры управления пайплайном:
1. `split.test_size`, `split.random_state`
2. `features.min_genre_frequency`
3. `training.n_estimators`, `training.max_depth`, `training.min_samples_leaf`

Пример сравнения эксперимента:

```bash
uv run dvc exp run -S training.n_estimators=260 -S training.max_depth=20
uv run dvc exp show --no-pager
```

## 6. `dvc push` и `dvc commit`

Загрузка данных в S3 remote:

```bash
uv run dvc push
```

Когда нужен `dvc commit`:
1. Когда выходные данные стадии обновились вне `dvc repro`.
2. Когда нужно зафиксировать новое состояние зависимостей/аутов в `dvc.lock`.

Пример:

```bash
uv run dvc commit prepare_dataset prepare_training_data train_popularity_model
git add dvc.lock
git commit -m "data: commit updated prepared dataset"
```

## 7. Запуск HTTP API сервиса предсказания популярности

Запуск:

```bash
uv run uvicorn src.service.app:app --reload --host 0.0.0.0 --port 8000
```

Проверка `health`:

```bash
curl http://127.0.0.1:8000/health
```

Пример запроса предсказания популярности:

```bash
curl -X POST http://127.0.0.1:8000/predict-popularity \
  -H "Content-Type: application/json" \
  -d '{
    "danceability": 0.72,
    "energy": 0.81,
    "valence": 0.54,
    "tempo": 122.0,
    "track_genre": "pop"
  }'
```

## 8. Переключение между версиями датасета

Переход на другую Git-ревизию:

```bash
git checkout <commit_or_tag>
uv run dvc checkout
```

Если нужных объектов нет в локальном кэше:

```bash
uv run dvc pull
uv run dvc checkout
```

## 9. Зачем версионировать данные

Основные моменты:

1. Можно воспроизвести любой эксперимент на исторической версии данных.
2. Понятно, какая версия данных использовалась для конкретной модели\версии сервиса.
3. Можно безопасно откатываться между версиями (`git checkout` + `dvc checkout`).
4. Данные хранятся в object storage, а в Git остаются легкие метафайлы.

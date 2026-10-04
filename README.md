# FrameCurator

Отбор качественных и разнообразных кадров из видео по попарным предпочтениям одного разметчика
с факторной проверкой «оценщик (готовая IQA / обученный) × отбор (top-k / MMR)».

Статус: **неделя 0** — скелет репозитория, симуляция мощности, черновик предрегистрации.
Спецификация и план — [CLAUDE.md](CLAUDE.md), протокол разметки — [docs/protocol.md](docs/protocol.md),
журнал — [docs/experiment_log.md](docs/experiment_log.md).

Предрегистрация: *ещё не опубликована* (черновик в [docs/preregistration.md](docs/preregistration.md)).

## Установка

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install        # или: pip install -r requirements.lock && pip install -e . --no-deps
make test lint
make power          # таблица мощности -> reports/power.csv
```

Инвентаризация видео (нужен `ffprobe` в PATH):

```bash
python scripts/inventory_videos.py /path/to/raw_videos --out data/meta/videos.csv
```

## Приватность

Сырые видео и кадры с людьми в репозиторий не попадают (`.gitignore`). Публикуются код, конфиги,
признаки без изображений лиц и метки.

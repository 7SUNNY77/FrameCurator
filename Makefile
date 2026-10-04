PYTHON ?= python

.PHONY: install test lint format power reproduce-small

install:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m ruff format --check src scripts tests

format:
	$(PYTHON) -m ruff format src scripts tests
	$(PYTHON) -m ruff check --fix src scripts tests

# Этап 0.2: симуляция мощности слепой оценки (раздел 7.5). Seed фиксирован в configs/default.yaml.
power:
	$(PYTHON) -m framecurator.eval.power --config configs/default.yaml --out reports/power.csv

# Весь конвейер на 3 видео с фикстурными метками (< 15 мин на CPU). Реализуется к этапу 1.6.
reproduce-small:
	@echo "reproduce-small: заглушка недели 0, конвейер появится на этапах 1.1-1.6"

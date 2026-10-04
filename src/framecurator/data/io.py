"""Ввод-вывод метаданных и меток.

Единственная точка чтения меток — :func:`load_labels`. Чтение test-меток
(раздел 2, п. 2 CLAUDE.md) разрешено только при ``FRAMECURATOR_FINAL_EVAL=1``
и указанном в ``FRAMECURATOR_FROZEN_TAG`` существующем git-теге замороженной модели.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pandas as pd

LABEL_FILES: dict[str, str] = {
    "trainval": "pairs_trainval.csv",
    "test": "pairs_test.csv",
}
FROZEN_TAG_RE = re.compile(r"^v\d+\.\d+(\.\d+)?-frozen$")


class TestAccessError(PermissionError):
    """Попытка прочитать test-метки без режима финальной оценки."""

    __test__ = False  # не путать pytest с тест-классом


def _git_tag_exists(tag: str, repo: Path) -> bool:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"],
            cwd=repo,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return res.returncode == 0


def check_test_access(repo: Path | None = None) -> str:
    """Проверить, что разрешено читать test-метки, и вернуть тег замороженной модели.

    Raises:
        TestAccessError: если не выставлен FRAMECURATOR_FINAL_EVAL=1, тег не задан,
            имеет неверный формат (ожидается ``vX.Y-frozen``) или отсутствует в git.
    """
    if os.environ.get("FRAMECURATOR_FINAL_EVAL") != "1":
        raise TestAccessError(
            "Чтение test-меток запрещено: нужен FRAMECURATOR_FINAL_EVAL=1 (раздел 2, п. 2)."
        )
    tag = os.environ.get("FRAMECURATOR_FROZEN_TAG", "")
    if not FROZEN_TAG_RE.match(tag):
        raise TestAccessError(f"FRAMECURATOR_FROZEN_TAG={tag!r}: ожидается тег вида vX.Y-frozen.")
    repo = repo or Path(__file__).resolve().parents[3]
    if not _git_tag_exists(tag, repo):
        raise TestAccessError(f"Git-тег {tag!r} не найден: модель не заморожена.")
    return tag


def load_labels(split: str, labels_dir: str | Path = "data/labels") -> pd.DataFrame:
    """Загрузить попарные метки сплита.

    Args:
        split: ``"trainval"`` или ``"test"``.
        labels_dir: каталог с CSV меток.

    Returns:
        DataFrame со схемой из раздела 6.5.

    Raises:
        TestAccessError: при ``split="test"`` вне режима финальной оценки.
        ValueError: при неизвестном сплите.
    """
    if split not in LABEL_FILES:
        raise ValueError(f"Неизвестный сплит {split!r}; допустимы {sorted(LABEL_FILES)}")
    if split == "test":
        check_test_access()
    return pd.read_csv(Path(labels_dir) / LABEL_FILES[split])

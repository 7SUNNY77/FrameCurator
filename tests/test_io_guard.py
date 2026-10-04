"""Защита test-меток (раздел 2, п. 2): load_labels("test") без режима финальной оценки падает."""

import pandas as pd
import pytest

from framecurator.data import io
from framecurator.data.io import TestAccessError, load_labels


@pytest.fixture
def labels_dir(tmp_path):
    # Синтетическая фикстура схемы, не человеческие метки
    df = pd.DataFrame(
        {"pair_id": [0], "video_id": ["v"], "frame_a": [1], "frame_b": [2], "choice": [1]}
    )
    df.to_csv(tmp_path / "pairs_trainval.csv", index=False)
    df.to_csv(tmp_path / "pairs_test.csv", index=False)
    return tmp_path


def test_trainval_readable(labels_dir, monkeypatch):
    monkeypatch.delenv("FRAMECURATOR_FINAL_EVAL", raising=False)
    assert len(load_labels("trainval", labels_dir)) == 1


def test_test_blocked_without_flag(labels_dir, monkeypatch):
    monkeypatch.delenv("FRAMECURATOR_FINAL_EVAL", raising=False)
    monkeypatch.setenv("FRAMECURATOR_FROZEN_TAG", "v1.0-frozen")
    with pytest.raises(TestAccessError):
        load_labels("test", labels_dir)


def test_test_blocked_without_tag(labels_dir, monkeypatch):
    monkeypatch.setenv("FRAMECURATOR_FINAL_EVAL", "1")
    monkeypatch.delenv("FRAMECURATOR_FROZEN_TAG", raising=False)
    with pytest.raises(TestAccessError):
        load_labels("test", labels_dir)


def test_test_blocked_bad_tag_format(labels_dir, monkeypatch):
    monkeypatch.setenv("FRAMECURATOR_FINAL_EVAL", "1")
    monkeypatch.setenv("FRAMECURATOR_FROZEN_TAG", "v0.0-skeleton")
    with pytest.raises(TestAccessError):
        load_labels("test", labels_dir)


def test_test_blocked_missing_git_tag(labels_dir, monkeypatch):
    monkeypatch.setenv("FRAMECURATOR_FINAL_EVAL", "1")
    monkeypatch.setenv("FRAMECURATOR_FROZEN_TAG", "v99.0-frozen")
    monkeypatch.setattr(io, "_git_tag_exists", lambda tag, repo: False)
    with pytest.raises(TestAccessError):
        load_labels("test", labels_dir)


def test_test_allowed_in_final_eval(labels_dir, monkeypatch):
    monkeypatch.setenv("FRAMECURATOR_FINAL_EVAL", "1")
    monkeypatch.setenv("FRAMECURATOR_FROZEN_TAG", "v1.0-frozen")
    monkeypatch.setattr(io, "_git_tag_exists", lambda tag, repo: True)
    assert len(load_labels("test", labels_dir)) == 1


def test_unknown_split(labels_dir):
    with pytest.raises(ValueError):
        load_labels("val", labels_dir)

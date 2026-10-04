"""Симуляция мощности слепой оценки H1 (этап 0.2, раздел 7.5).

Модель данных: истинная доля побед Mm над B6 на видео ``p_v ~ N(p_true, sd_video)``
(обрезается в [0.01, 0.99]); на каждом видео ``n_raters`` независимых оценщиков,
доля побед на видео — ``Binomial(n_raters, p_v) / n_raters``. H1 подтверждена, если
нижняя граница перцентильного bootstrap-CI по видео (уровень ``1 - alpha``) > 0.5
и средняя доля побед >= ``min_point``.

``alpha = 0.05`` воспроизводит код из 7.5. Порог H1 в 9.2 требует поправки Холма
для семейства из 3 гипотез: в худшем случае H1 проверяется на уровне ``0.05 / 3``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml


def simulate_h1(
    n_videos: int,
    n_raters: int,
    p_true: float,
    sd_video: float = 0.15,
    n_sim: int = 2000,
    n_boot: int = 2000,
    alphas: Sequence[float] = (0.05,),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Смоделировать ``n_sim`` слепых оценок и вернуть статистики для решения по H1.

    Порядок вызовов генератора совпадает с эталонным кодом из раздела 7.5,
    поэтому при ``alphas=(0.05,)`` результат идентичен ему.

    Returns:
        means: массив ``(n_sim,)`` средних долей побед по видео.
        lowers: массив ``(n_sim, len(alphas))`` нижних границ двустороннего
            bootstrap-CI уровня ``1 - alpha``.
    """
    rng = np.random.default_rng(seed)
    qs = np.asarray(alphas, dtype=float) / 2
    means = np.empty(n_sim)
    lowers = np.empty((n_sim, len(qs)))
    for i in range(n_sim):
        pv = np.clip(rng.normal(p_true, sd_video, n_videos), 0.01, 0.99)
        wins = rng.binomial(n_raters, pv) / n_raters
        boots = rng.choice(wins, size=(n_boot, n_videos)).mean(axis=1)
        means[i] = wins.mean()
        lowers[i] = np.quantile(boots, qs)
    return means, lowers


def power_sim(
    n_videos: int = 20,
    n_raters: int = 5,
    p_true: float = 0.65,
    sd_video: float = 0.15,
    n_sim: int = 2000,
    n_boot: int = 2000,
    min_point: float = 0.60,
    seed: int = 0,
    alpha: float = 0.05,
) -> float:
    """Вероятность подтвердить H1 при истинной доле побед ``p_true`` (раздел 7.5)."""
    means, lowers = simulate_h1(n_videos, n_raters, p_true, sd_video, n_sim, n_boot, (alpha,), seed)
    return float(np.mean((lowers[:, 0] > 0.5) & (means >= min_point)))


def power_grid(
    n_videos: Sequence[int],
    n_raters: Sequence[int],
    p_true: Sequence[float],
    min_point: Sequence[float],
    alpha: Sequence[float],
    sd_video: float = 0.15,
    n_sim: int = 2000,
    n_boot: int = 2000,
    seed: int = 0,
) -> list[dict[str, float | int]]:
    """Перебор сетки параметров. Один seed на все ячейки (общие случайные числа).

    Для каждой тройки (n_videos, n_raters, p_true) симуляция выполняется один раз,
    а все комбинации (min_point, alpha) оцениваются на одних и тех же данных.
    """
    rows: list[dict[str, float | int]] = []
    for nv, nr, p in itertools.product(n_videos, n_raters, p_true):
        means, lowers = simulate_h1(nv, nr, p, sd_video, n_sim, n_boot, alpha, seed)
        for j, a in enumerate(alpha):
            for mp in min_point:
                pw = float(np.mean((lowers[:, j] > 0.5) & (means >= mp)))
                rows.append(
                    {
                        "n_videos": nv,
                        "n_raters": nr,
                        "p_true": p,
                        "min_point": mp,
                        "alpha": round(a, 6),
                        "power": round(pw, 4),
                        "mc_se": round(float(np.sqrt(pw * (1 - pw) / n_sim)), 4),
                        "sd_video": sd_video,
                        "n_sim": n_sim,
                        "n_boot": n_boot,
                        "seed": seed,
                    }
                )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: перебор сетки из ``configs/default.yaml`` и запись в ``reports/power.csv``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="reports/power.csv")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pc = cfg["power"]
    rows = power_grid(
        n_videos=pc["n_videos"],
        n_raters=pc["n_raters"],
        p_true=pc["p_true"],
        min_point=pc["min_point"],
        alpha=pc["alpha"],
        sd_video=pc["sd_video"],
        n_sim=pc["n_sim"],
        n_boot=pc["n_boot"],
        seed=cfg["seed"],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} строк -> {out}")


if __name__ == "__main__":
    main()

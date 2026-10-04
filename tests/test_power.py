"""Тесты симуляции мощности (раздел 7.5)."""

import numpy as np

from framecurator.eval.power import power_grid, power_sim, simulate_h1


def reference_power_sim(
    n_videos=20, n_raters=5, p_true=0.65, sd_video=0.15, n_sim=2000, n_boot=2000,
    min_point=0.60, seed=0,
):  # fmt: skip
    """Дословная копия кода из раздела 7.5 CLAUDE.md."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        pv = np.clip(rng.normal(p_true, sd_video, n_videos), 0.01, 0.99)
        wins = rng.binomial(n_raters, pv) / n_raters
        boots = rng.choice(wins, size=(n_boot, n_videos)).mean(axis=1)
        hits += (np.quantile(boots, 0.025) > 0.5) and (wins.mean() >= min_point)
    return hits / n_sim


def test_matches_reference():
    kw = dict(n_videos=15, n_raters=3, p_true=0.65, n_sim=200, n_boot=300, seed=7)
    assert power_sim(**kw) == reference_power_sim(**kw)


def test_deterministic():
    kw = dict(n_videos=10, n_raters=3, p_true=0.6, n_sim=100, n_boot=200, seed=1)
    assert power_sim(**kw) == power_sim(**kw)


def test_null_rarely_confirms():
    # При p_true = 0.5 и min_point = 0.5 доля ложных подтверждений около alpha/2 (односторонне)
    pw = power_sim(n_videos=20, n_raters=5, p_true=0.5, n_sim=1000, n_boot=500, min_point=0.5)
    assert pw < 0.06


def test_strong_effect_high_power():
    pw = power_sim(n_videos=30, n_raters=7, p_true=0.8, n_sim=300, n_boot=500)
    assert pw > 0.95


def test_stricter_alpha_lowers_power():
    means, lowers = simulate_h1(20, 5, 0.65, n_sim=300, n_boot=500, alphas=(0.05, 0.05 / 3))
    assert np.all(lowers[:, 1] <= lowers[:, 0])


def test_grid_shape():
    rows = power_grid([10, 20], [3], [0.6, 0.7], [0.6, 0.55], [0.05], n_sim=50, n_boot=100)
    assert len(rows) == 2 * 1 * 2 * 2 * 1
    assert all(0.0 <= r["power"] <= 1.0 for r in rows)

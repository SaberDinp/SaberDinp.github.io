"""Tests for elite count resolution (integer vs optional elite_fraction)."""

from solver.ga import _resolve_elite_size


def test_elite_size_plain():
    assert _resolve_elite_size({"elite_size": 4}, 1000) == 4


def test_elite_fraction_overrides():
    cfg = {"elite_size": 4, "elite_fraction": 0.01}
    assert _resolve_elite_size(cfg, 2000) == 20
    cfg2 = {"elite_size": 4, "elite_fraction": 0.004}
    assert _resolve_elite_size(cfg2, 1000) == 4


def test_elite_clamped_to_pop_minus_one():
    assert _resolve_elite_size({"elite_size": 9999}, 100) == 99


def test_invalid_fraction_falls_back_to_elite_size():
    cfg = {"elite_size": 6, "elite_fraction": 4.0}
    assert _resolve_elite_size(cfg, 1000) == 6

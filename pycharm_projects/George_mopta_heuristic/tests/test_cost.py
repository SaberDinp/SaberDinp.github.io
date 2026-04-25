"""Tests for cost.py — lane costs, difficult-lane surcharge, total cost."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from solver.cost import (
    truck_lane_earnings,
    truck_dedicated_cost,
    total_solution_cost,
)


# ── truck_lane_earnings ───────────────────────────────────────────

def test_truck_lane_earnings_empty():
    base = np.array([100.0, 200.0, 300.0])
    assert truck_lane_earnings([], base) == pytest.approx(0.0)


def test_truck_lane_earnings_all():
    base = np.array([100.0, 200.0, 300.0])
    assert truck_lane_earnings([0, 1, 2], base) == pytest.approx(600.0)


def test_truck_lane_earnings_subset():
    base = np.array([100.0, 500.0, 300.0])
    assert truck_lane_earnings([0, 2], base) == pytest.approx(400.0)


# ── truck_dedicated_cost ──────────────────────────────────────────

def test_truck_dedicated_cost_above_minimum():
    base = np.array([1500.0, 2000.0, 1200.0])
    cost = truck_dedicated_cost([0, 1, 2], base, min_weekly_earnings=4000.0)
    assert cost == pytest.approx(4700.0)  # 4700 > 4000


def test_truck_dedicated_cost_below_minimum():
    base = np.array([500.0, 1000.0])
    cost = truck_dedicated_cost([0, 1], base, min_weekly_earnings=4000.0)
    assert cost == pytest.approx(4000.0)  # guaranteed minimum


def test_truck_dedicated_cost_exactly_minimum():
    base = np.array([2000.0, 2000.0])
    cost = truck_dedicated_cost([0, 1], base, min_weekly_earnings=4000.0)
    assert cost == pytest.approx(4000.0)


def test_truck_dedicated_cost_no_orders():
    base = np.array([1000.0, 2000.0])
    cost = truck_dedicated_cost([], base, min_weekly_earnings=4000.0)
    assert cost == pytest.approx(0.0)


# ── total_solution_cost ───────────────────────────────────────────

def _make_arrays(n_orders):
    """Create deterministic arrays for testing."""
    rng = np.random.default_rng(0)
    base = rng.uniform(500, 3000, size=n_orders)
    surcharge = np.zeros(n_orders)
    surcharge[0] = 200.0  # order 0 has a surcharge
    surcharge[2] = 150.0  # order 2 has a surcharge
    return base, surcharge


def test_total_cost_all_tender():
    n = 5
    base, surcharge = _make_arrays(n)
    chrom = np.full(n, -1, dtype=np.int8)
    total, ded, tend = total_solution_cost(chrom, base, surcharge, n_trucks=3)
    # All tendered: cost = sum(base + surcharge)
    expected = float(np.sum(base + surcharge))
    assert total == pytest.approx(expected)
    assert ded == pytest.approx(0.0)
    assert tend == pytest.approx(expected)


def test_total_cost_one_truck_all_orders_above_min():
    n = 3
    base = np.array([2000.0, 2000.0, 2000.0])
    surcharge = np.zeros(n)
    chrom = np.zeros(n, dtype=np.int8)  # all to truck 0
    total, ded, tend = total_solution_cost(chrom, base, surcharge, n_trucks=3, min_weekly_earnings=4000.0)
    assert ded == pytest.approx(6000.0)  # 6000 > 4000, so no minimum bump
    assert tend == pytest.approx(0.0)
    assert total == pytest.approx(6000.0)


def test_total_cost_one_truck_below_min():
    n = 2
    base = np.array([1000.0, 1000.0])
    surcharge = np.zeros(n)
    chrom = np.array([0, 0], dtype=np.int8)
    total, ded, tend = total_solution_cost(chrom, base, surcharge, n_trucks=3, min_weekly_earnings=4000.0)
    assert ded == pytest.approx(4000.0)  # minimum kicks in
    assert tend == pytest.approx(0.0)


def test_total_cost_mixed_tender_surcharge():
    """Tender orders with surcharge should cost more than tender without."""
    base = np.array([1000.0, 1000.0])
    surcharge = np.array([0.0, 200.0])
    chrom = np.array([-1, -1], dtype=np.int8)
    total, _, tend = total_solution_cost(chrom, base, surcharge, n_trucks=3)
    assert total == pytest.approx(2200.0)


def test_surcharge_not_applied_to_dedicated():
    """Dedicated orders must NOT incur the surcharge."""
    base = np.array([1000.0, 1000.0])
    surcharge = np.array([500.0, 500.0])  # both orders have surcharge
    # Assign both to truck 0 (dedicated)
    chrom = np.array([0, 0], dtype=np.int8)
    total, ded, tend = total_solution_cost(chrom, base, surcharge, n_trucks=3, min_weekly_earnings=0.0)
    # Dedicated cost = sum(base) only = 2000, no surcharge
    assert ded == pytest.approx(2000.0)
    assert tend == pytest.approx(0.0)
    # If tendered instead, cost = sum(base + surcharge) = 3000
    chrom_tender = np.array([-1, -1], dtype=np.int8)
    total_t, _, tend_t = total_solution_cost(chrom_tender, base, surcharge, n_trucks=3, min_weekly_earnings=0.0)
    assert tend_t == pytest.approx(3000.0)
    assert ded < tend_t  # dedicated cheaper on difficult lanes


def test_inactive_truck_contributes_zero():
    """Trucks with no assigned orders should contribute $0 (not $4,000)."""
    n = 2
    base = np.array([2000.0, 2000.0])
    surcharge = np.zeros(n)
    # Only truck 0 is used
    chrom = np.array([0, 0], dtype=np.int8)
    _, ded, _ = total_solution_cost(chrom, base, surcharge, n_trucks=5, min_weekly_earnings=4000.0)
    # Truck 0: 4000 > 4000 no, 4000 == 4000 -> 4000
    assert ded == pytest.approx(4000.0)  # one active truck at minimum

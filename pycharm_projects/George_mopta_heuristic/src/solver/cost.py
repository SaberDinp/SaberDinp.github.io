"""
cost.py — Freight cost computation for dedicated and tendered orders.

Key rules (from John Cox + problem PDF):
  - Dedicated driver cost to Interfor = Lane.Cost (baseline, same as open market)
    No difficult-lane surcharge for dedicated fleet.
  - Tender cost = Lane.Cost + (DifficultLane.adder_per_mile * Lane.Mileage)
    Difficult-lane surcharge applies ONLY to tendered orders.
  - Dedicated truck minimum guarantee: Interfor pays max(sum_lane_costs, $4,000/week).
    This is modeled as a cost, not just a hard constraint.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from .data import Order, Truck, ProblemData
from .geography import find_difficult_lane_adder


# ──────────────────────────────────────────────────────────────────
# Per-order tender surcharge (cached at startup)
# ──────────────────────────────────────────────────────────────────

def build_tender_surcharge_array(problem: ProblemData) -> np.ndarray:
    """
    Return float64 array of shape (n_orders,) giving the difficult-lane
    surcharge for each order IF tendered.

    surcharge[i] = adder_per_mile * mileage(origin_i, dest_i)
    """
    n = len(problem.orders)
    surcharges = np.zeros(n, dtype=np.float64)
    for i, order in enumerate(problem.orders):
        adder = find_difficult_lane_adder(
            order.origin, order.destination, problem.difficult_lanes
        )
        if adder > 0:
            mileage = problem.lane_mileage.get((order.origin, order.destination), 0.0)
            surcharges[i] = adder * mileage
    return surcharges


def build_order_base_cost_array(problem: ProblemData) -> np.ndarray:
    """Return float64 array of shape (n_orders,) with Lane.Cost for each order."""
    return np.array(
        [problem.lane_cost.get((o.origin, o.destination), 0.0) for o in problem.orders],
        dtype=np.float64,
    )


# ──────────────────────────────────────────────────────────────────
# Truck-level cost helpers (non-vectorized, for readable feasibility)
# ──────────────────────────────────────────────────────────────────

def truck_lane_earnings(
    order_indices: List[int],
    order_base_cost: np.ndarray,
) -> float:
    """Sum of loaded lane costs for a truck's assigned orders."""
    if not order_indices:
        return 0.0
    return float(np.sum(order_base_cost[order_indices]))


def truck_dedicated_cost(
    order_indices: List[int],
    order_base_cost: np.ndarray,
    min_weekly_earnings: float = 4000.0,
) -> float:
    """
    Cost to Interfor for one dedicated truck per week.
    = max(sum_of_lane_costs, min_weekly_earnings)
    Returns 0 if the truck has no assigned orders (not activated).
    """
    if not order_indices:
        return 0.0
    earnings = truck_lane_earnings(order_indices, order_base_cost)
    return max(earnings, min_weekly_earnings)


# ──────────────────────────────────────────────────────────────────
# Solution-level cost (used in GA fitness and final reporting)
# ──────────────────────────────────────────────────────────────────

def total_solution_cost(
    assignment: np.ndarray,
    order_base_cost: np.ndarray,
    tender_surcharge: np.ndarray,
    n_trucks: int,
    min_weekly_earnings: float = 4000.0,
) -> Tuple[float, float, float]:
    """
    Compute total freight cost for an assignment chromosome.

    Parameters
    ----------
    assignment : int array (n_orders,), values in {-1 (tender), 0..n_trucks-1}
    order_base_cost : base lane cost per order
    tender_surcharge : difficult-lane surcharge per order (applied only if tendered)
    n_trucks : number of dedicated trucks
    min_weekly_earnings : $4,000 guarantee per active truck

    Returns
    -------
    (total_cost, dedicated_cost, tender_cost)
    """
    # ── Tender cost ──────────────────────────────────────────────
    tender_mask = assignment == -1
    tender_cost = float(np.sum(
        order_base_cost[tender_mask] + tender_surcharge[tender_mask]
    ))

    # ── Dedicated cost ───────────────────────────────────────────
    dedicated_cost = 0.0
    for t in range(n_trucks):
        mask = assignment == t
        if not np.any(mask):
            continue
        earnings = float(np.sum(order_base_cost[mask]))
        dedicated_cost += max(earnings, min_weekly_earnings)

    return dedicated_cost + tender_cost, dedicated_cost, tender_cost

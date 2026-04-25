"""
feasibility.py — Constraint checker for a truck's weekly schedule.

Checks all five hard constraints from the competition PDF:
  1. Drivers must earn >= $4,000/week.
  2. Final load delivered within 100 miles of home.
  3. >= 2 consecutive days off per week (only after delivery + return home).
  4. Max 450 miles/day.
  5. Max 50% of weekly miles can be deadhead.

All functions return a list of (constraint_name, violation_description) tuples.
Empty list = fully feasible.

Also provides a SoftPenalty calculator used by the GA fitness function.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .data import ProblemData, Truck
from .scheduler import TruckSchedule
from .cost import truck_lane_earnings
from .geography import city_distance_miles


Violations = List[Tuple[str, str]]


# ──────────────────────────────────────────────────────────────────
# Hard constraint checks
# ──────────────────────────────────────────────────────────────────

def check_min_earnings(
    schedule: TruckSchedule,
    order_base_cost: np.ndarray,
    order_index_map: dict,        # order_id -> position in order_base_cost
    min_earnings: float = 4000.0,
) -> Violations:
    order_indices = [
        order_index_map[leg.order_id]
        for leg in schedule.legs
        if leg.leg_type == "loaded" and leg.order_id is not None
    ]
    # Deduplicate (multi-day legs emit multiple Leg records for same order)
    order_indices = list(dict.fromkeys(order_indices))
    earnings = truck_lane_earnings(order_indices, order_base_cost)
    if earnings < min_earnings:
        return [("min_earnings",
                 f"earnings ${earnings:.2f} < ${min_earnings:.2f} minimum")]
    return []


def check_final_delivery_distance(
    schedule: TruckSchedule,
    truck: Truck,
    problem: ProblemData,
    max_dist: float = 100.0,
) -> Violations:
    # No work done → trivially at home
    if schedule.final_delivery_location == truck.home:
        return []
    dist = city_distance_miles(
        schedule.final_delivery_location, truck.home, problem
    )
    if dist > max_dist:
        return [("final_delivery_distance",
                 f"{schedule.final_delivery_location} -> home {dist:.1f} mi > {max_dist:.1f} mi")]
    return []


def check_daily_miles(
    schedule: TruckSchedule,
    max_per_day: float = 450.0,
) -> Violations:
    """
    Each calendar day must have total miles <= max_per_day.
    (The scheduler already enforces this during construction, so this is a
    double-check / assertion guard.)
    """
    day_miles: dict[int, float] = {}
    for leg in schedule.legs:
        day_miles[leg.day] = day_miles.get(leg.day, 0.0) + leg.miles
    violations = []
    for day, miles in day_miles.items():
        if miles > max_per_day + 1e-6:
            violations.append(("daily_miles",
                                f"day {day}: {miles:.1f} mi > {max_per_day:.1f} mi cap"))
    return violations


def check_deadhead_fraction(
    schedule: TruckSchedule,
    max_fraction: float = 0.50,
) -> Violations:
    if schedule.total_miles < 1e-9:
        return []
    frac = schedule.deadhead_miles / schedule.total_miles
    if frac > max_fraction + 1e-6:
        return [("deadhead_fraction",
                 f"deadhead {schedule.deadhead_miles:.1f} / total {schedule.total_miles:.1f}"
                 f" = {frac:.1%} > {max_fraction:.0%}")]
    return []


def check_working_days(
    schedule: TruckSchedule,
    max_working_days: int = 5,
) -> Violations:
    """
    Driver must have >= 2 consecutive days off.
    With a 7-day week this means working_days <= 5.
    """
    if schedule.working_days > max_working_days:
        return [("working_days",
                 f"route needs {schedule.working_days} days > {max_working_days} max working days")]
    return []


def check_all(
    schedule: TruckSchedule,
    truck: Truck,
    problem: ProblemData,
    order_base_cost: np.ndarray,
    order_index_map: dict,
    cfg_constraints: dict,
) -> Violations:
    """Run all five constraint checks. Returns combined violations list."""
    c = cfg_constraints
    viols = []
    viols += check_min_earnings(schedule, order_base_cost, order_index_map,
                                c.get("min_weekly_earnings", 4000.0))
    viols += check_final_delivery_distance(schedule, truck, problem,
                                           c.get("max_final_dist_from_home", 100.0))
    viols += check_daily_miles(schedule, c.get("max_miles_per_day", 450.0))
    viols += check_deadhead_fraction(schedule, c.get("max_deadhead_fraction", 0.50))
    viols += check_working_days(schedule, c.get("max_working_days", 5))
    return viols


# ──────────────────────────────────────────────────────────────────
# Soft penalty (for GA fitness during evolution)
# ──────────────────────────────────────────────────────────────────

def soft_penalty(
    schedule: TruckSchedule,
    truck: Truck,
    problem: ProblemData,
    cfg_constraints: dict,
    cfg_penalties: dict,
) -> float:
    """
    Compute a non-negative soft penalty for near-feasible constraint violations.
    Used during GA evolution only — the final reported solution is hard-checked.
    """
    penalty = 0.0
    c = cfg_constraints
    p = cfg_penalties
    min_earn = c.get("min_weekly_earnings", 4000.0)

    if schedule.route:
        earn = float(
            sum(
                problem.lane_cost.get((o.origin, o.destination), 0.0)
                for o in schedule.route
            )
        )
        short = max(0.0, min_earn - earn)
        penalty += short * p.get("min_weekly_earnings_shortfall", 3.0)

        dist = city_distance_miles(
            schedule.final_delivery_location, truck.home, problem
        )
        overage = max(0.0, dist - c.get("max_final_dist_from_home", 100.0))
        penalty += overage * p.get("final_dist_from_home", 5.0)

    # Deadhead fraction
    if schedule.total_miles > 1e-9:
        frac = schedule.deadhead_miles / schedule.total_miles
        max_frac = c.get("max_deadhead_fraction", 0.50)
        overage = max(0.0, frac - max_frac)
        penalty += overage * p.get("deadhead_fraction", 2000.0)

    # Total miles over 5-day cap
    cap_5day = c.get("max_working_days", 5) * c.get("max_miles_per_day", 450.0)
    miles_over = max(0.0, schedule.total_miles - cap_5day)
    penalty += miles_over * p.get("total_miles_over_5days", 3.0)

    return penalty


# ──────────────────────────────────────────────────────────────────
# Repair operator: eject orders that cause hard violations
# ──────────────────────────────────────────────────────────────────

def repair_truck_route(
    truck: Truck,
    order_indices: list[int],
    problem: ProblemData,
    order_base_cost: np.ndarray,
    cfg_constraints: dict,
) -> tuple[list[int], list[int]]:
    """
    Greedily adjust a truck's assignment until ``check_all`` passes (or the
    truck is emptied to tender).

    Ejection policy:
      * ``min_earnings``: cannot raise the sum of lane costs by removing loads,
        so **tender all** orders on this truck.
      * ``final_delivery_distance``: remove the **last** order in the current
        NN route (changes where the return-home leg starts).
      * Miles / deadhead / working days: remove the **cheapest** lane-cost order.

    Returns (kept_indices, ejected_indices) using **global** order indices.
    """
    from .scheduler import build_nearest_neighbor_route, schedule_truck_week

    max_mpd = cfg_constraints.get("max_miles_per_day", 450.0)
    max_miles = cfg_constraints.get("max_working_days", 5) * max_mpd
    max_dh_frac = cfg_constraints.get("max_deadhead_fraction", 0.50)
    min_earn = cfg_constraints.get("min_weekly_earnings", 4000.0)

    order_index_map = {o.order_id: o.order_idx for o in problem.orders}

    kept = list(order_indices)
    ejected: list[int] = []
    max_iter = max(len(order_indices) * 4 + 8, 24)

    for _ in range(max_iter):
        if not kept:
            break

        route = build_nearest_neighbor_route(
            truck, np.array(kept, dtype=np.int64), problem
        )
        sched = schedule_truck_week(truck, route, problem, max_mpd)

        viols = check_all(
            sched, truck, problem, order_base_cost,
            order_index_map, cfg_constraints,
        )

        miles_ok = sched.total_miles <= max_miles + 1e-6
        dh_ok = (
            sched.total_miles < 1e-9
            or sched.deadhead_miles / sched.total_miles <= max_dh_frac + 1e-6
        )

        if not viols:
            break

        codes = {v[0] for v in viols}

        if "min_earnings" in codes:
            total_earn = truck_lane_earnings(kept, order_base_cost)
            if total_earn < min_earn - 1e-6:
                ejected.extend(kept)
                kept = []
                break

        if "final_delivery_distance" in codes:
            last_gidx = route[-1].order_idx
            kept.remove(last_gidx)
            ejected.append(last_gidx)
            continue

        if (
            not miles_ok
            or "deadhead_fraction" in codes
            or "daily_miles" in codes
            or "working_days" in codes
        ):
            cheapest_idx = min(kept, key=lambda i: float(order_base_cost[i]))
            ejected.append(cheapest_idx)
            kept.remove(cheapest_idx)
            continue

        # Unexpected mix (e.g. numerical mismatch): fall back to cheapest eject
        cheapest_idx = min(kept, key=lambda i: float(order_base_cost[i]))
        ejected.append(cheapest_idx)
        kept.remove(cheapest_idx)

    return kept, ejected

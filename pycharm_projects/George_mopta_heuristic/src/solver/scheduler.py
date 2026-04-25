"""
scheduler.py — Route scheduling for a single truck within one planning week.

Responsibilities:
  1. Build a nearest-neighbor route for a set of assigned orders.
  2. Assign each driving leg to a calendar day (greedy bin-packing, 450 mi/day).
  3. Return loaded miles, deadhead miles, working days used, final-delivery location.

Terminology:
  loaded leg   : order.origin -> order.destination  (revenue-generating)
  deadhead leg : previous location -> order.origin  (empty driving)
  return leg   : last delivery -> home              (deadhead, end of week)

Week model:
  Max 5 working days (Mon-Fri), then >= 2 consecutive days off (Sat-Sun).
  Constraint checked in feasibility.py, not enforced here.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .data import Order, Truck, ProblemData


# ──────────────────────────────────────────────────────────────────
# Output types
# ──────────────────────────────────────────────────────────────────

@dataclass
class Leg:
    """One driving segment assigned to a calendar day."""
    day: int
    leg_type: str          # "deadhead" | "loaded" | "return_home"
    origin: str
    destination: str
    miles: float
    order_id: Optional[int] = None  # set for loaded legs


@dataclass
class TruckSchedule:
    """Full weekly schedule for one truck."""
    truck_id: int
    route: List[Order]          # ordered list of shipments
    legs: List[Leg]
    loaded_miles: float
    deadhead_miles: float
    total_miles: float
    working_days: int           # last calendar day with driving
    final_delivery_location: str


# ──────────────────────────────────────────────────────────────────
# Nearest-neighbour route builder
# ──────────────────────────────────────────────────────────────────

def build_nearest_neighbor_route(
    truck: Truck,
    order_indices: np.ndarray,      # indices into problem.orders
    problem: ProblemData,
) -> List[Order]:
    """
    Greedy nearest-neighbor: starting from truck home, repeatedly pick the
    unserved order whose ORIGIN is closest (by mileage) to the current location.

    Returns ordered list of Order objects.
    """
    if len(order_indices) == 0:
        return []

    mileage_matrix = problem.mileage_matrix
    loc_idx = problem.loc_idx
    orders = problem.orders

    home_idx = loc_idx.get(truck.home)
    if home_idx is None:
        # Fallback: return arbitrary order
        return [orders[i] for i in order_indices]

    order_orig_idx = np.array([loc_idx.get(orders[i].origin, 0) for i in order_indices])
    order_dest_idx = np.array([loc_idx.get(orders[i].destination, 0) for i in order_indices])

    n = len(order_indices)
    visited = np.zeros(n, dtype=bool)
    route_positions = np.empty(n, dtype=int)
    current_loc_idx = home_idx

    for step in range(n):
        # Distances from current_loc to each unvisited order's origin
        dists = mileage_matrix[current_loc_idx, order_orig_idx]
        dists[visited] = np.inf
        best = int(np.argmin(dists))
        visited[best] = True
        route_positions[step] = best
        current_loc_idx = order_dest_idx[best]

    return [orders[order_indices[pos]] for pos in route_positions]


# ──────────────────────────────────────────────────────────────────
# Day-assignment scheduler
# ──────────────────────────────────────────────────────────────────

def schedule_truck_week(
    truck: Truck,
    route: List[Order],
    problem: ProblemData,
    max_miles_per_day: float = 450.0,
) -> TruckSchedule:
    """
    Assign loaded and deadhead legs to calendar days.

    Algorithm: greedy sequential — fill the current day up to max_miles_per_day,
    then overflow to the next day. A single leg may span multiple days.
    """
    if not route:
        return TruckSchedule(
            truck_id=truck.truck_id,
            route=[],
            legs=[],
            loaded_miles=0.0,
            deadhead_miles=0.0,
            total_miles=0.0,
            working_days=0,
            final_delivery_location=truck.home,
        )

    lane_mileage = problem.lane_mileage

    def get_miles(a: str, b: str) -> float:
        return lane_mileage.get((a, b), 0.0)

    # Build raw leg sequence: (type, from, to, miles, order_id)
    raw_legs: List[Tuple[str, str, str, float, Optional[int]]] = []
    current_loc = truck.home
    for order in route:
        dh = get_miles(current_loc, order.origin)
        if dh > 0 or current_loc != order.origin:
            raw_legs.append(("deadhead", current_loc, order.origin, dh, None))
        ld = get_miles(order.origin, order.destination)
        raw_legs.append(("loaded", order.origin, order.destination, ld, order.order_id))
        current_loc = order.destination
    # Return home
    ret = get_miles(current_loc, truck.home)
    raw_legs.append(("return_home", current_loc, truck.home, ret, None))
    final_delivery_location = route[-1].destination

    # Assign days
    legs: List[Leg] = []
    current_day = 1
    miles_remaining_today = max_miles_per_day
    loaded_miles = 0.0
    deadhead_miles = 0.0

    for leg_type, frm, to, total_leg_miles, oid in raw_legs:
        remaining = total_leg_miles
        while True:
            if remaining <= 0.0:
                break
            can_drive = min(miles_remaining_today, remaining)
            legs.append(Leg(
                day=current_day,
                leg_type=leg_type,
                origin=frm,
                destination=to,
                miles=can_drive,
                order_id=oid,
            ))
            remaining -= can_drive
            miles_remaining_today -= can_drive
            if miles_remaining_today < 1e-9:
                current_day += 1
                miles_remaining_today = max_miles_per_day

        if leg_type == "loaded":
            loaded_miles += total_leg_miles
        else:
            deadhead_miles += total_leg_miles

    # Handle zero-mile legs (same origin == dest): still record them
    for raw in raw_legs:
        if raw[3] == 0.0 and raw[0] == "loaded":
            loaded_miles += 0.0  # already counted

    # working_days = last day that had any driving
    working_days = legs[-1].day if legs else 0

    return TruckSchedule(
        truck_id=truck.truck_id,
        route=route,
        legs=legs,
        loaded_miles=loaded_miles,
        deadhead_miles=deadhead_miles,
        total_miles=loaded_miles + deadhead_miles,
        working_days=working_days,
        final_delivery_location=final_delivery_location,
    )

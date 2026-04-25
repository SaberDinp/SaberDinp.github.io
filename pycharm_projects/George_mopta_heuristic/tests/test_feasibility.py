"""
Tests for feasibility.py — constraint checks on truck schedules.

Uses a toy problem (3 locations, 1 truck, 2-3 orders) with known
distances so expected violations are computed by hand.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from solver.data import Order, Truck, ProblemData
from solver.scheduler import TruckSchedule, Leg, build_nearest_neighbor_route, schedule_truck_week
from solver.feasibility import (
    check_daily_miles,
    check_deadhead_fraction,
    check_working_days,
    check_final_delivery_distance,
    check_min_earnings,
    repair_truck_route,
)


# ── Toy problem fixture ───────────────────────────────────────────

def make_toy_problem():
    """
    3 locations: HOME, A, B.
    HOME -> A: 200 mi, cost 800
    A    -> B: 100 mi, cost 400
    B -> HOME: 50  mi (within 100 mi home proximity)
    HOME -> B: 250 mi
    A -> HOME: 250 mi
    B -> A: 100 mi
    All other pairs: symmetric.
    """
    locs = {
        "HOME,GA": (33.0, -84.0),
        "A,GA":    (34.0, -84.0),
        "B,GA":    (34.5, -84.0),
    }
    lane_mileage = {
        ("HOME,GA", "A,GA"):   200.0,
        ("A,GA",    "HOME,GA"): 200.0,
        ("A,GA",    "B,GA"):   100.0,
        ("B,GA",    "A,GA"):   100.0,
        ("HOME,GA", "B,GA"):   250.0,
        ("B,GA",    "HOME,GA"):  50.0,
        ("HOME,GA", "HOME,GA"):   0.0,
        ("A,GA",    "A,GA"):      0.0,
        ("B,GA",    "B,GA"):      0.0,
    }
    lane_cost = {k: v * 4.0 for k, v in lane_mileage.items()}  # $4/mi

    orders = [
        Order(order_id=1, origin="HOME,GA", destination="A,GA",
              due_date=None, week_label="test", order_idx=0),
        Order(order_id=2, origin="A,GA", destination="B,GA",
              due_date=None, week_label="test", order_idx=1),
    ]

    truck = Truck(truck_id=0, name="TestTruck", home="HOME,GA",
                  home_lat=33.0, home_lon=-84.0)

    problem = ProblemData(
        orders=orders,
        trucks=[truck],
        locations=locs,
        lane_cost=lane_cost,
        lane_mileage=lane_mileage,
        difficult_lanes=[],
        orders_by_week={"test": orders},
    )
    problem.build_arrays()
    return problem, truck


def make_toy_schedule(truck, route, problem, max_mpd=450):
    return schedule_truck_week(truck, route, problem, max_miles_per_day=max_mpd)


# ── check_daily_miles ─────────────────────────────────────────────

class TestDailyMiles:
    def test_no_violation(self):
        # Each leg well below 450 mi
        truck = Truck(0, "T", "HOME,GA", 33.0, -84.0)
        problem, truck = make_toy_problem()
        # Route: order1 then order2
        route = [problem.orders[0], problem.orders[1]]
        sched = make_toy_schedule(truck, route, problem, max_mpd=450)
        viols = check_daily_miles(sched, max_per_day=450)
        assert viols == []

    def test_violation_when_cap_tiny(self):
        problem, truck = make_toy_problem()
        route = [problem.orders[0], problem.orders[1]]
        sched = make_toy_schedule(truck, route, problem, max_mpd=450)
        # Artificially check with smaller cap
        viols = check_daily_miles(sched, max_per_day=50)
        assert len(viols) > 0
        assert all(v[0] == "daily_miles" for v in viols)


# ── check_deadhead_fraction ───────────────────────────────────────

class TestDeadheadFraction:
    def test_no_deadhead_no_violation(self):
        sched = TruckSchedule(
            truck_id=0, route=[], legs=[],
            loaded_miles=1000.0, deadhead_miles=0.0,
            total_miles=1000.0, working_days=2,
            final_delivery_location="A,GA",
        )
        assert check_deadhead_fraction(sched, max_fraction=0.50) == []

    def test_exactly_50_pct_no_violation(self):
        sched = TruckSchedule(
            truck_id=0, route=[], legs=[],
            loaded_miles=500.0, deadhead_miles=500.0,
            total_miles=1000.0, working_days=3,
            final_delivery_location="A,GA",
        )
        assert check_deadhead_fraction(sched, max_fraction=0.50) == []

    def test_over_50_pct_violation(self):
        sched = TruckSchedule(
            truck_id=0, route=[], legs=[],
            loaded_miles=400.0, deadhead_miles=600.0,
            total_miles=1000.0, working_days=3,
            final_delivery_location="A,GA",
        )
        viols = check_deadhead_fraction(sched, max_fraction=0.50)
        assert len(viols) == 1
        assert viols[0][0] == "deadhead_fraction"

    def test_zero_total_miles_no_violation(self):
        sched = TruckSchedule(
            truck_id=0, route=[], legs=[],
            loaded_miles=0.0, deadhead_miles=0.0,
            total_miles=0.0, working_days=0,
            final_delivery_location="HOME,GA",
        )
        assert check_deadhead_fraction(sched) == []


# ── check_working_days ────────────────────────────────────────────

class TestWorkingDays:
    def test_5_days_no_violation(self):
        sched = TruckSchedule(0, [], [], 0, 0, 0, working_days=5,
                              final_delivery_location="A,GA")
        assert check_working_days(sched, max_working_days=5) == []

    def test_6_days_violation(self):
        sched = TruckSchedule(0, [], [], 0, 0, 0, working_days=6,
                              final_delivery_location="A,GA")
        viols = check_working_days(sched, max_working_days=5)
        assert len(viols) == 1
        assert viols[0][0] == "working_days"


# ── check_final_delivery_distance ─────────────────────────────────

class TestFinalDeliveryDistance:
    def test_within_100_miles(self):
        problem, truck = make_toy_problem()
        # B,GA -> HOME,GA is 50 miles (within 100)
        sched = TruckSchedule(0, [], [], 100, 50, 150, 1,
                              final_delivery_location="B,GA")
        viols = check_final_delivery_distance(sched, truck, problem, max_dist=100.0)
        assert viols == []

    def test_over_100_miles(self):
        problem, truck = make_toy_problem()
        # A,GA -> HOME,GA is 200 miles (over 100)
        sched = TruckSchedule(0, [], [], 200, 50, 250, 1,
                              final_delivery_location="A,GA")
        viols = check_final_delivery_distance(sched, truck, problem, max_dist=100.0)
        assert len(viols) == 1
        assert viols[0][0] == "final_delivery_distance"

    def test_empty_route_no_violation(self):
        problem, truck = make_toy_problem()
        sched = TruckSchedule(0, [], [], 0, 0, 0, 0,
                              final_delivery_location="HOME,GA")
        viols = check_final_delivery_distance(sched, truck, problem, max_dist=100.0)
        assert viols == []


# ── check_min_earnings ────────────────────────────────────────────

class TestMinEarnings:
    def _make_sched_with_orders(self, order_ids):
        """Create a fake schedule whose loaded legs reference given order IDs."""
        legs = [
            Leg(day=1, leg_type="loaded", origin="A", destination="B",
                miles=100.0, order_id=oid)
            for oid in order_ids
        ]
        return TruckSchedule(0, [], legs, 100.0, 0.0, 100.0, 1, "B")

    def test_above_minimum(self):
        base_cost = np.array([2000.0, 3000.0, 1000.0])
        order_index_map = {10: 0, 20: 1, 30: 2}
        sched = self._make_sched_with_orders([10, 20])  # earnings = 5000
        viols = check_min_earnings(sched, base_cost, order_index_map, min_earnings=4000.0)
        assert viols == []

    def test_below_minimum(self):
        base_cost = np.array([1000.0, 500.0])
        order_index_map = {10: 0, 20: 1}
        sched = self._make_sched_with_orders([10, 20])  # earnings = 1500
        viols = check_min_earnings(sched, base_cost, order_index_map, min_earnings=4000.0)
        assert len(viols) == 1
        assert viols[0][0] == "min_earnings"


# ── Integration: full schedule for toy problem ────────────────────

def make_far_home_problem():
    """
    One order HOME -> FAR where return FAR -> HOME exceeds 100 mi cap.
    Repair should eject that order (tender entire truck load).
    """
    locs = {
        "HOME,GA": (33.0, -84.0),
        "FAR,GA": (35.0, -90.0),
    }
    lane_mileage = {
        ("HOME,GA", "FAR,GA"): 400.0,
        ("FAR,GA", "HOME,GA"): 150.0,
        ("HOME,GA", "HOME,GA"): 0.0,
        ("FAR,GA", "FAR,GA"): 0.0,
    }
    lane_cost = {k: v * 10.0 for k, v in lane_mileage.items()}

    orders = [
        Order(
            order_id=1,
            origin="HOME,GA",
            destination="FAR,GA",
            due_date=None,
            week_label="test",
            order_idx=0,
        ),
    ]
    truck = Truck(truck_id=0, name="Solo", home="HOME,GA", home_lat=33.0, home_lon=-84.0)
    problem = ProblemData(
        orders=orders,
        trucks=[truck],
        locations=locs,
        lane_cost=lane_cost,
        lane_mileage=lane_mileage,
        difficult_lanes=[],
        orders_by_week={"test": orders},
    )
    problem.build_arrays()
    return problem, truck


def test_repair_ejects_when_final_return_exceeds_cap():
    problem, truck = make_far_home_problem()
    base = np.array([problem.lane_cost[("HOME,GA", "FAR,GA")]], dtype=np.float64)
    cfg = {
        "max_miles_per_day": 450.0,
        "max_working_days": 5,
        "max_deadhead_fraction": 0.50,
        "min_weekly_earnings": 4000.0,
        "max_final_dist_from_home": 100.0,
    }
    kept, ejected = repair_truck_route(
        truck, [0], problem, base, cfg
    )
    assert kept == []
    assert ejected == [0]


def test_repair_tenders_all_when_below_min_earnings():
    problem, truck = make_toy_problem()
    # Two cheap orders: lane costs 800 + 400 = 1200 < 4000
    base = np.array([800.0, 400.0], dtype=np.float64)
    cfg = {
        "max_miles_per_day": 450.0,
        "max_working_days": 5,
        "max_deadhead_fraction": 0.50,
        "min_weekly_earnings": 4000.0,
        "max_final_dist_from_home": 100.0,
    }
    kept, ejected = repair_truck_route(
        truck, [0, 1], problem, base, cfg
    )
    assert kept == []
    assert set(ejected) == {0, 1}


def test_toy_problem_feasible_schedule():
    """Full end-to-end: build route, schedule, check feasibility."""
    problem, truck = make_toy_problem()
    order_indices = np.array([0, 1])
    route = build_nearest_neighbor_route(truck, order_indices, problem)
    assert len(route) == 2

    sched = schedule_truck_week(truck, route, problem, max_miles_per_day=450)
    # Total miles: loaded (HOME->A=200, A->B=100) + deadhead (B->HOME=50) = 350
    # All fits in 1 day (350 < 450)
    assert sched.working_days == 1
    assert sched.loaded_miles == pytest.approx(300.0)   # 200 + 100
    assert sched.deadhead_miles == pytest.approx(50.0)  # return B->HOME only (HOME->A is first leg)

    # Check constraints pass
    viols_dh = check_deadhead_fraction(sched, 0.50)
    assert viols_dh == [], f"Unexpected deadhead violation: {viols_dh}"

    viols_days = check_working_days(sched, 5)
    assert viols_days == []

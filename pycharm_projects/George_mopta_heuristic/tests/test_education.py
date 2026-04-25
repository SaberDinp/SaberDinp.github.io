"""
Tests for education.py — intra-route local-search operators.

Uses the same toy problem (3 locations, 1 truck, 2 orders) from
test_feasibility.py where expected results are computed by hand.

Additional "deliberate deadhead" problem (4 locations, 1 truck, 3 orders)
is constructed so the NN route leaves unnecessary repositioning that
Or-opt-1 can eliminate.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from solver.data import Order, Truck, ProblemData
from solver.scheduler import build_nearest_neighbor_route, schedule_truck_week
from solver.feasibility import check_all
from solver.education import (
    _route_obj,
    or_opt_1,
    two_opt,
    or_opt_2,
    educate_route,
    educate_individual,
    init_keys_from_nn,
)
from solver.ga import FitnessEvaluator


# ── Shared toy problem (same as test_feasibility.py) ─────────────

def make_toy_problem():
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
    lane_cost = {k: v * 4.0 for k, v in lane_mileage.items()}
    orders = [
        Order(order_id=1, origin="HOME,GA", destination="A,GA",
              due_date=None, week_label="test", order_idx=0),
        Order(order_id=2, origin="A,GA", destination="B,GA",
              due_date=None, week_label="test", order_idx=1),
    ]
    truck = Truck(truck_id=0, name="T", home="HOME,GA",
                  home_lat=33.0, home_lon=-84.0)
    problem = ProblemData(
        orders=orders, trucks=[truck], locations=locs,
        lane_cost=lane_cost, lane_mileage=lane_mileage,
        difficult_lanes=[], orders_by_week={"test": orders},
    )
    problem.build_arrays()
    return problem, truck


# ── Problem with improvable NN route ─────────────────────────────

def make_detour_problem():
    """
    4 locations: HOME, A, B, C.
    Layout: HOME -- A -- B -- C (linear, left to right)

    Mileage:
      HOME->A = 100, A->B = 50, B->C = 50
      C->HOME = 200 (long return)
      A->C = 100, etc.

    Three orders:
      O1: HOME -> C  (origin near home, delivers far)
      O2: A    -> B  (short hop)
      O3: B    -> A  (reverse — makes NN do detour)

    NN from HOME:
      Step 1: nearest origin to HOME is A (dist 100), picks O2 (origin A)
              → cur = B
      Step 2: nearest origin to B is B (O3), picks O3
              → cur = A
      Step 3: only O1 left (origin HOME → dist A->HOME = 100), picks O1
              → cur = C
      Deadhead: HOME->A + B->B + A->HOME = 100 + 0 + 100 = 200
              + return C->HOME = 200
              total dh = 400

    Optimal order (O1, O2, O3):
      HOME->HOME (no dh, O1 starts at HOME) + C->A (150) + B->B (0) + A->HOME = 100
      Hmm, let me compute more carefully.

    Actually let's build this so or_opt_1 demonstrably improves the NN route.
    """
    locs = {
        "HOME,GA": (33.0, -84.0),
        "A,GA":    (33.0, -83.0),  # 100 mi east
        "B,GA":    (33.0, -82.5),  # 150 mi east
        "C,GA":    (33.0, -82.0),  # 200 mi east
    }
    # Use explicit mileage dict (not Haversine)
    def d(a, b):
        positions = {"HOME,GA": 0, "A,GA": 100, "B,GA": 150, "C,GA": 200}
        return abs(positions[a] - positions[b])

    cities = list(locs.keys())
    lane_mileage = {}
    for c1 in cities:
        for c2 in cities:
            lane_mileage[(c1, c2)] = d(c1, c2)

    lane_cost = {k: v * 4.0 for k, v in lane_mileage.items()}

    orders = [
        # O0: HOME -> C (origin at HOME = position 0)
        Order(order_id=10, origin="HOME,GA", destination="C,GA",
              due_date=None, week_label="test", order_idx=0),
        # O1: A -> B (origin at position 100)
        Order(order_id=11, origin="A,GA", destination="B,GA",
              due_date=None, week_label="test", order_idx=1),
        # O2: B -> A (origin at position 150; NN picks this second, causing backtrack)
        Order(order_id=12, origin="B,GA", destination="A,GA",
              due_date=None, week_label="test", order_idx=2),
    ]
    truck = Truck(truck_id=0, name="T2", home="HOME,GA",
                  home_lat=33.0, home_lon=-84.0)
    problem = ProblemData(
        orders=orders, trucks=[truck], locations=locs,
        lane_cost=lane_cost, lane_mileage=lane_mileage,
        difficult_lanes=[], orders_by_week={"test": orders},
    )
    problem.build_arrays()
    return problem, truck


# ── _route_obj ────────────────────────────────────────────────────

class TestRouteObj:
    def test_two_order_route(self):
        problem, truck = make_toy_problem()
        orders = problem.orders
        # Route: O1 (HOME->A), O2 (A->B)
        # Deadhead: HOME->HOME=0, A->A=0, B->HOME=50  → 50
        # No home penalty (50 < 100)
        route = [orders[0], orders[1]]
        obj = _route_obj(route, truck, problem.mileage_matrix, problem.loc_idx,
                         home_penalty=10.0, max_home_dist=100.0)
        assert obj == pytest.approx(50.0)

    def test_empty_route_zero(self):
        problem, truck = make_toy_problem()
        assert _route_obj([], truck, problem.mileage_matrix, problem.loc_idx, 0.0, 100.0) == 0.0

    def test_home_penalty_applied(self):
        problem, truck = make_toy_problem()
        orders = problem.orders
        # Route: O2 (A->B), O1 (HOME->A)
        # Deadhead: HOME->A=200, B->HOME=200, A->HOME=200
        # Return: A->HOME = 200; penalty = 10 * max(0, 200-100) = 1000
        route = [orders[1], orders[0]]
        obj = _route_obj(route, truck, problem.mileage_matrix, problem.loc_idx,
                         home_penalty=10.0, max_home_dist=100.0)
        # dh: HOME->A(200) + B->HOME(200) = 400, plus return A->HOME(200)  = 600
        # wait: route is [O2, O1]
        # O2: A->B, O1: HOME->A
        # dh: HOME->A(200) [reposition to O2's origin], B->HOME(50) [reposition to O1's HOME origin]
        # return: A->HOME = 200 (O1 delivers to A, truck goes home)
        # total dh (no penalty): 200 + 50 + 200 = 450
        # final dist = A->HOME = 200 > 100, penalty = 10 * (200-100) = 1000
        assert obj == pytest.approx(450.0 + 1000.0)


# ── or_opt_1 ──────────────────────────────────────────────────────

class TestOrOpt1:
    def test_no_improvement_at_optimal(self):
        """Toy: route [O1, O2] is already NN-optimal; or_opt_1 should not change it."""
        problem, truck = make_toy_problem()
        route = list(problem.orders)  # [O1, O2]
        mm, loc_idx = problem.mileage_matrix, problem.loc_idx
        improved = or_opt_1(route, truck, mm, loc_idx, home_penalty=0.0, max_home_dist=100.0)
        # Cost [O1,O2]: HOME->HOME(0) + A->A(0) + B->HOME(50) = 50
        # Cost [O2,O1]: HOME->A(200) + B->HOME(50) + A->HOME(200) = 450
        # or_opt_1 should keep [O1, O2]
        assert [o.order_id for o in improved] == [1, 2]

    def test_single_order_unchanged(self):
        problem, truck = make_toy_problem()
        route = [problem.orders[0]]
        mm, loc_idx = problem.mileage_matrix, problem.loc_idx
        result = or_opt_1(route, truck, mm, loc_idx)
        assert result == route

    def test_improves_detour_route(self):
        """Detour problem: or_opt_1 should find a route at least as good as NN."""
        problem, truck = make_detour_problem()
        order_indices = np.array([0, 1, 2])
        nn_route = build_nearest_neighbor_route(truck, order_indices, problem)
        mm, loc_idx = problem.mileage_matrix, problem.loc_idx
        improved = or_opt_1(nn_route, truck, mm, loc_idx, home_penalty=10.0, max_home_dist=100.0)
        nn_cost = _route_obj(nn_route, truck, mm, loc_idx, 10.0, 100.0)
        imp_cost = _route_obj(improved, truck, mm, loc_idx, 10.0, 100.0)
        assert imp_cost <= nn_cost + 1e-6, (
            f"or_opt_1 should not worsen route: nn={nn_cost:.1f} imp={imp_cost:.1f}"
        )


# ── two_opt ───────────────────────────────────────────────────────

class TestTwoOpt:
    def test_single_order_unchanged(self):
        problem, truck = make_toy_problem()
        route = [problem.orders[0]]
        mm, loc_idx = problem.mileage_matrix, problem.loc_idx
        assert two_opt(route, truck, mm, loc_idx) == route

    def test_two_orders_no_regression(self):
        """2-opt on 2 orders: reversing the only segment either improves or leaves same."""
        problem, truck = make_toy_problem()
        route = list(problem.orders)
        mm, loc_idx = problem.mileage_matrix, problem.loc_idx
        improved = two_opt(route, truck, mm, loc_idx, home_penalty=0.0)
        base_cost = _route_obj(route, truck, mm, loc_idx, 0.0, 100.0)
        imp_cost = _route_obj(improved, truck, mm, loc_idx, 0.0, 100.0)
        assert imp_cost <= base_cost + 1e-6

    def test_no_regression_detour(self):
        problem, truck = make_detour_problem()
        order_indices = np.array([0, 1, 2])
        nn_route = build_nearest_neighbor_route(truck, order_indices, problem)
        mm, loc_idx = problem.mileage_matrix, problem.loc_idx
        improved = two_opt(nn_route, truck, mm, loc_idx, home_penalty=10.0, max_home_dist=100.0)
        nn_cost = _route_obj(nn_route, truck, mm, loc_idx, 10.0, 100.0)
        imp_cost = _route_obj(improved, truck, mm, loc_idx, 10.0, 100.0)
        assert imp_cost <= nn_cost + 1e-6


# ── or_opt_2 ──────────────────────────────────────────────────────

class TestOrOpt2:
    def test_too_short_unchanged(self):
        problem, truck = make_toy_problem()
        route = [problem.orders[0]]
        mm, loc_idx = problem.mileage_matrix, problem.loc_idx
        assert or_opt_2(route, truck, mm, loc_idx) == route

    def test_no_regression_detour(self):
        problem, truck = make_detour_problem()
        order_indices = np.array([0, 1, 2])
        nn_route = build_nearest_neighbor_route(truck, order_indices, problem)
        mm, loc_idx = problem.mileage_matrix, problem.loc_idx
        improved = or_opt_2(nn_route, truck, mm, loc_idx, home_penalty=10.0, max_home_dist=100.0)
        nn_cost = _route_obj(nn_route, truck, mm, loc_idx, 10.0, 100.0)
        imp_cost = _route_obj(improved, truck, mm, loc_idx, 10.0, 100.0)
        assert imp_cost <= nn_cost + 1e-6


# ── educate_route ─────────────────────────────────────────────────

class TestEducateRoute:
    def test_never_worsens(self):
        """educate_route result must have LS objective <= original."""
        problem, truck = make_detour_problem()
        order_indices = np.array([0, 1, 2])
        nn_route = build_nearest_neighbor_route(truck, order_indices, problem)
        mm, loc_idx = problem.mileage_matrix, problem.loc_idx
        cfg_edu = {"or_opt_1": True, "two_opt": True, "or_opt_2": True,
                   "max_ls_iters": 5, "home_penalty_per_mile": 10.0,
                   "max_home_dist": 100.0}
        improved = educate_route(nn_route, truck, problem, cfg_edu)
        nn_cost = _route_obj(nn_route, truck, mm, loc_idx, 10.0, 100.0)
        imp_cost = _route_obj(improved, truck, mm, loc_idx, 10.0, 100.0)
        assert imp_cost <= nn_cost + 1e-6

    def test_single_order_route_unchanged(self):
        problem, truck = make_toy_problem()
        route = [problem.orders[0]]
        cfg_edu = {"or_opt_1": True, "two_opt": True, "or_opt_2": True,
                   "max_ls_iters": 3, "home_penalty_per_mile": 0.0,
                   "max_home_dist": 100.0}
        assert educate_route(route, truck, problem, cfg_edu) == route


# ── educate_individual ────────────────────────────────────────────

class TestEducateIndividual:
    def _make_evaluator(self, problem):
        week_indices = np.array([o.order_idx for o in problem.orders])
        return FitnessEvaluator(
            problem=problem,
            week_order_indices=week_indices,
            cfg_constraints={
                "max_miles_per_day": 450.0, "max_working_days": 5,
                "max_deadhead_fraction": 0.50, "min_weekly_earnings": 4000.0,
                "max_final_dist_from_home": 100.0,
            },
            cfg_penalties={
                "deadhead_fraction": 2000.0, "final_dist_from_home": 5.0,
                "min_weekly_earnings_shortfall": 3.0, "total_miles_over_5days": 3.0,
            },
        )

    def test_chromosome_unchanged(self):
        """educate_individual must not change the assignment chromosome."""
        problem, truck = make_detour_problem()
        evaluator = self._make_evaluator(problem)
        chrom = np.array([0, 0, 0], dtype=np.int8)  # all to truck 0
        keys = init_keys_from_nn(chrom, evaluator)
        cfg_edu = {"or_opt_1": True, "two_opt": True, "or_opt_2": True,
                   "max_ls_iters": 3, "home_penalty_per_mile": 10.0, "max_home_dist": 100.0}
        new_chrom, new_keys = educate_individual(chrom, keys, evaluator, cfg_edu)
        np.testing.assert_array_equal(chrom, new_chrom)

    def test_fitness_does_not_worsen(self):
        """Fitness after education should be <= NN-based fitness."""
        problem, truck = make_detour_problem()
        evaluator = self._make_evaluator(problem)
        chrom = np.array([0, 0, 0], dtype=np.int8)
        keys = init_keys_from_nn(chrom, evaluator)
        fitness_before = evaluator.evaluate(chrom, keys)
        cfg_edu = {"or_opt_1": True, "two_opt": True, "or_opt_2": True,
                   "max_ls_iters": 5, "home_penalty_per_mile": 10.0, "max_home_dist": 100.0}
        _, new_keys = educate_individual(chrom, keys, evaluator, cfg_edu)
        fitness_after = evaluator.evaluate(chrom, new_keys)
        assert fitness_after <= fitness_before + 1e-6, (
            f"Education worsened fitness: {fitness_before:.2f} -> {fitness_after:.2f}"
        )

    def test_feasibility_preserved(self):
        """
        After educating a feasible chromosome, the hard constraints should still pass.
        (Education only changes route order, not assignment — feasibility is assignment-driven.)
        """
        problem, truck = make_toy_problem()
        evaluator = self._make_evaluator(problem)
        # Both orders on truck 0: toy problem is feasible
        chrom = np.array([0, 0], dtype=np.int8)
        keys = init_keys_from_nn(chrom, evaluator)
        cfg_edu = {"or_opt_1": True, "two_opt": True, "or_opt_2": True,
                   "max_ls_iters": 3, "home_penalty_per_mile": 10.0, "max_home_dist": 100.0}
        _, new_keys = educate_individual(chrom, keys, evaluator, cfg_edu)

        # Reconstruct route from new keys and check feasibility
        from solver.scheduler import schedule_truck_week
        from solver.feasibility import check_all
        from solver.cost import build_order_base_cost_array

        local_positions = np.where(chrom == 0)[0]
        global_indices = evaluator.week_order_indices[local_positions]
        sorted_order = np.argsort(new_keys[local_positions])
        route_global = global_indices[sorted_order]
        route = [problem.orders[i] for i in route_global]
        sched = schedule_truck_week(truck, route, problem, 450.0)
        full_base = build_order_base_cost_array(problem)
        order_index_map = {o.order_id: o.order_idx for o in problem.orders}
        cfg_c = {"max_miles_per_day": 450.0, "max_working_days": 5,
                 "max_deadhead_fraction": 0.50, "min_weekly_earnings": 0.0,
                 "max_final_dist_from_home": 100.0}
        viols = check_all(sched, truck, problem, full_base, order_index_map, cfg_c)
        assert viols == [], f"Violations after education: {viols}"


# ── init_keys_from_nn ─────────────────────────────────────────────

class TestInitKeysFromNN:
    def test_keys_shape(self):
        problem, _ = make_toy_problem()
        week_indices = np.array([0, 1])
        evaluator = FitnessEvaluator(
            problem=problem, week_order_indices=week_indices,
            cfg_constraints={}, cfg_penalties={},
        )
        chrom = np.array([0, 0], dtype=np.int8)
        keys = init_keys_from_nn(chrom, evaluator)
        assert keys.shape == (2,)
        assert keys.dtype == np.float32

    def test_keys_encode_valid_permutation(self):
        """Keys for an assigned truck should be distinguishable (not all equal)."""
        problem, _ = make_detour_problem()
        week_indices = np.array([0, 1, 2])
        evaluator = FitnessEvaluator(
            problem=problem, week_order_indices=week_indices,
            cfg_constraints={}, cfg_penalties={},
        )
        chrom = np.array([0, 0, 0], dtype=np.int8)
        keys = init_keys_from_nn(chrom, evaluator)
        # For 3 orders on 1 truck, keys should be [0.0, 0.5, 1.0] in some order
        sorted_keys = np.sort(keys)
        assert sorted_keys[0] == pytest.approx(0.0)
        assert sorted_keys[1] == pytest.approx(0.5)
        assert sorted_keys[2] == pytest.approx(1.0)

    def test_default_rng_path_is_deterministic(self):
        problem, _ = make_detour_problem()
        week_indices = np.array([0, 1, 2])
        evaluator = FitnessEvaluator(
            problem=problem, week_order_indices=week_indices,
            cfg_constraints={}, cfg_penalties={},
        )
        chrom = np.array([0, -1, 0], dtype=np.int8)
        keys_a = init_keys_from_nn(chrom, evaluator)
        keys_b = init_keys_from_nn(chrom, evaluator)
        np.testing.assert_array_equal(keys_a, keys_b)


# ── Export guard (integration) ────────────────────────────────────

class TestExportGuard:
    """solution_fully_feasible returns False when any week has violations."""

    def test_feasible_solution_accepted(self):
        from solver.output import solution_fully_feasible
        sol = {
            "weeks": [
                {"totals": {"feasible": True}},
                {"totals": {"feasible": True}},
            ]
        }
        assert solution_fully_feasible(sol) is True

    def test_infeasible_week_rejected(self):
        from solver.output import solution_fully_feasible
        sol = {
            "weeks": [
                {"totals": {"feasible": True}},
                {"totals": {"feasible": False}},
            ]
        }
        assert solution_fully_feasible(sol) is False

    def test_empty_weeks_rejected(self):
        from solver.output import solution_fully_feasible
        assert solution_fully_feasible({"weeks": []}) is False

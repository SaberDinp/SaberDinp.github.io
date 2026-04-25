"""
tests/test_data.py — Integration test: load real workbook and validate structure.

Skipped if the Excel file is not found (CI without data).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np

DATA_PATH = Path(__file__).parent.parent / "AIMMS_MOPTA_2026" / "AIMMS-MOPTA Interfor data.xlsx"
pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="Excel workbook not found — skipping integration tests"
)


@pytest.fixture(scope="module")
def problem():
    from solver.data import load_problem
    return load_problem(DATA_PATH)


def test_locations_count(problem):
    assert len(problem.locations) == 218


def test_trucks_count(problem):
    assert len(problem.trucks) == 20


def test_orders_count(problem):
    assert len(problem.orders) == 1000


def test_two_due_date_weeks(problem):
    assert set(problem.orders_by_week.keys()) == {"2026-01-17", "2026-01-24"}


def test_lane_count(problem):
    # 218 * 218 = 47,524
    assert len(problem.lane_cost) == 218 * 218


def test_difficult_lanes_count(problem):
    assert len(problem.difficult_lanes) == 61


def test_all_truck_homes_in_locations(problem):
    bad = [t for t in problem.trucks if t.home not in problem.locations]
    assert bad == [], f"Truck homes not in Locations: {[t.home for t in bad]}"


def test_all_order_origins_in_locations(problem):
    bad = [o for o in problem.orders if o.origin not in problem.locations]
    assert bad == [], f"{len(bad)} orders have origins not in Locations"


def test_all_order_destinations_in_locations(problem):
    bad = [o for o in problem.orders if o.destination not in problem.locations]
    assert bad == [], f"{len(bad)} orders have destinations not in Locations"


def test_all_order_lanes_exist(problem):
    missing = [
        o for o in problem.orders
        if (o.origin, o.destination) not in problem.lane_cost
    ]
    assert missing == [], f"{len(missing)} orders lack a lane entry"


def test_cost_matrix_no_nan_for_valid_pairs(problem):
    """All lane-table pairs should be non-NaN in the matrix."""
    assert problem.cost_matrix is not None
    nan_count = int(np.isnan(problem.cost_matrix).sum())
    # 218*218 = 47524 pairs all in lanes table, so matrix should be fully populated
    assert nan_count == 0, f"{nan_count} NaN entries in cost matrix"


def test_lane_self_loops_have_zero_mileage(problem):
    """Self-loop lanes (origin=dest) should have 0 mileage."""
    zero_count = sum(
        1 for (o, d), m in problem.lane_mileage.items()
        if o == d and m == 0.0
    )
    # 218 self-loops
    assert zero_count == 218


def test_difficult_lane_origins_in_locations(problem):
    """Difficult lane origins should all be valid city names."""
    dl_origins = {dl.origin for dl in problem.difficult_lanes}
    bad = dl_origins - set(problem.locations.keys())
    assert bad == set(), f"Difficult lane origins not in Locations: {bad}"


def test_difficult_lane_surcharge_positive(problem):
    for dl in problem.difficult_lanes:
        assert dl.adder_per_mile > 0, f"Non-positive adder for {dl}"


def test_nearest_neighbor_route_covers_all_orders(problem):
    """Nearest-neighbor route for first truck should return all assigned orders."""
    from solver.scheduler import build_nearest_neighbor_route
    truck = problem.trucks[0]
    # Use first 5 orders in week 2026-01-17
    week_orders = problem.orders_by_week["2026-01-17"][:5]
    indices = np.array([o.order_idx for o in week_orders])
    route = build_nearest_neighbor_route(truck, indices, problem)
    assert len(route) == 5
    assert {o.order_id for o in route} == {o.order_id for o in week_orders}


def test_schedule_truck_week_day_cap(problem):
    """Schedule should never exceed 450 miles on any single day."""
    from solver.scheduler import build_nearest_neighbor_route, schedule_truck_week
    truck = problem.trucks[0]
    # Assign 10 orders to one truck — enough to span multiple days
    week_orders = problem.orders_by_week["2026-01-17"][:10]
    indices = np.array([o.order_idx for o in week_orders])
    route = build_nearest_neighbor_route(truck, indices, problem)
    sched = schedule_truck_week(truck, route, problem, max_miles_per_day=450)

    from collections import defaultdict
    day_miles = defaultdict(float)
    for leg in sched.legs:
        day_miles[leg.day] += leg.miles
    for day, miles in day_miles.items():
        assert miles <= 450 + 1e-6, f"Day {day} has {miles:.1f} miles > 450"


def test_build_tender_surcharge_array_shape(problem):
    from solver.cost import build_tender_surcharge_array
    arr = build_tender_surcharge_array(problem)
    assert arr.shape == (1000,)
    assert (arr >= 0).all()
    # At least some orders should have non-zero surcharge
    assert arr.sum() > 0


def test_base_cost_array_all_positive(problem):
    from solver.cost import build_order_base_cost_array
    arr = build_order_base_cost_array(problem)
    # All orders should have a lane entry after city-name normalization.
    # The minimum lane cost in the data is $400 (self-loops), so every entry >= 400.
    assert (arr >= 400).all(), (
        f"Some orders have cost < $400 (likely missing lane after normalization). "
        f"Min cost: {arr.min():.2f}, zero count: {(arr == 0).sum()}"
    )

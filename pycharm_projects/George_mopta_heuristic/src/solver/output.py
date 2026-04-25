"""
output.py — Write solution.json and solution_summary.csv.

Output contract (competition requirement):
  Per carrier: ordered list of loads (order IDs), day, leg type, miles,
               loaded vs empty, freight cost for each leg.
  Tendered orders: lane, baseline cost + surcharge.
  Totals: freight by mode, deadhead %, constraint slacks.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .data import ProblemData, Order, Truck
from .scheduler import build_nearest_neighbor_route, schedule_truck_week, TruckSchedule
from .cost import build_order_base_cost_array, build_tender_surcharge_array
from .feasibility import check_all

logger = logging.getLogger(__name__)


def build_week_solution(
    week_label: str,
    best_chrom: np.ndarray,
    week_order_indices: np.ndarray,
    problem: ProblemData,
    cfg_constraints: dict,
) -> dict:
    """
    Decode a chromosome into a full week solution dict.
    Returns a JSON-serializable dict.
    """
    n_trucks = len(problem.trucks)
    full_base_cost = build_order_base_cost_array(problem)
    full_surcharge = build_tender_surcharge_array(problem)
    base_cost = full_base_cost[week_order_indices]
    surcharge = full_surcharge[week_order_indices]

    # order_id -> global order_idx (for indexing full_base_cost / full_surcharge arrays)
    local_orders = [problem.orders[i] for i in week_order_indices]
    order_index_map = {o.order_id: o.order_idx for o in local_orders}

    week_sol: dict = {
        "week": week_label,
        "carriers": [],
        "tendered_orders": [],
        "totals": {},
    }

    total_dedicated_cost = 0.0
    total_tender_cost = 0.0
    all_violations: List[dict] = []

    # ── Dedicated carriers ───────────────────────────────────────
    for t_id in range(n_trucks):
        local_positions = np.where(best_chrom == t_id)[0]
        if len(local_positions) == 0:
            continue

        # Chromosome uses local positions; convert to global problem.orders indices
        global_indices = week_order_indices[local_positions]

        truck = problem.trucks[t_id]
        route = build_nearest_neighbor_route(truck, global_indices, problem)
        sched = schedule_truck_week(
            truck, route, problem,
            cfg_constraints.get("max_miles_per_day", 450.0)
        )

        earnings = float(np.sum(base_cost[local_positions]))
        truck_cost = max(earnings, cfg_constraints.get("min_weekly_earnings", 4000.0))
        total_dedicated_cost += truck_cost

        # Feasibility check
        viols = check_all(
            sched, truck, problem,
            full_base_cost, order_index_map, cfg_constraints
        )
        if viols:
            all_violations.append({
                "truck": truck.name,
                "violations": [{"constraint": c, "detail": d} for c, d in viols],
            })

        # Build legs output
        legs_out = []
        for leg in sched.legs:
            legs_out.append({
                "day": leg.day,
                "type": leg.leg_type,
                "from": leg.origin,
                "to": leg.destination,
                "miles": round(leg.miles, 2),
                "order_id": leg.order_id,
            })

        week_sol["carriers"].append({
            "carrier_id": t_id,
            "carrier_name": truck.name,
            "home": truck.home,
            "orders": [o.order_id for o in route],
            "legs": legs_out,
            "loaded_miles": round(sched.loaded_miles, 2),
            "deadhead_miles": round(sched.deadhead_miles, 2),
            "total_miles": round(sched.total_miles, 2),
            "deadhead_pct": round(
                100.0 * sched.deadhead_miles / sched.total_miles
                if sched.total_miles > 0 else 0.0, 1
            ),
            "working_days": sched.working_days,
            "earnings": round(earnings, 2),
            "truck_cost": round(truck_cost, 2),
            "feasible": len(viols) == 0,
            "violations": [{"constraint": c, "detail": d} for c, d in viols],
        })

    # ── Tendered orders ──────────────────────────────────────────
    tender_local_positions = np.where(best_chrom == -1)[0]
    for lp in tender_local_positions:
        order = local_orders[lp]
        bc = float(base_cost[lp])
        sur = float(surcharge[lp])
        total_tender_cost += bc + sur
        week_sol["tendered_orders"].append({
            "order_id": order.order_id,
            "origin": order.origin,
            "destination": order.destination,
            "base_cost": round(bc, 2),
            "surcharge": round(sur, 2),
            "total_cost": round(bc + sur, 2),
        })

    grand_total = total_dedicated_cost + total_tender_cost
    week_sol["totals"] = {
        "dedicated_cost": round(total_dedicated_cost, 2),
        "tender_cost": round(total_tender_cost, 2),
        "grand_total": round(grand_total, 2),
        "n_dedicated_orders": int(len(week_order_indices) - len(tender_local_positions)),
        "n_tendered_orders": int(len(tender_local_positions)),
        "n_active_trucks": int(sum(1 for t in range(n_trucks) if np.any(best_chrom == t))),
        "feasible": len(all_violations) == 0,
        "constraint_violations": all_violations,
    }

    return week_sol


def write_solution_json(solution: dict, output_dir: Path, filename: str = "solution.json") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(solution, f, indent=2, default=str)
    logger.info("Wrote %s", path)
    return path


def write_summary_csv(solution: dict, output_dir: Path, filename: str = "solution_summary.csv") -> Path:
    """Flat CSV: one row per (week, carrier/tender, order)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    rows = []
    for week_data in solution.get("weeks", []):
        week = week_data["week"]
        for carrier in week_data.get("carriers", []):
            for oid in carrier["orders"]:
                rows.append({
                    "week": week,
                    "mode": "dedicated",
                    "carrier_name": carrier["carrier_name"],
                    "carrier_id": carrier["carrier_id"],
                    "order_id": oid,
                    "loaded_miles": carrier["loaded_miles"],
                    "deadhead_miles": carrier["deadhead_miles"],
                    "deadhead_pct": carrier["deadhead_pct"],
                    "truck_cost": carrier["truck_cost"],
                    "working_days": carrier["working_days"],
                    "feasible": carrier["feasible"],
                })
        for order in week_data.get("tendered_orders", []):
            rows.append({
                "week": week,
                "mode": "tender",
                "carrier_name": "open_market",
                "carrier_id": -1,
                "order_id": order["order_id"],
                "loaded_miles": "",
                "deadhead_miles": "",
                "deadhead_pct": "",
                "truck_cost": order["total_cost"],
                "working_days": "",
                "feasible": True,
            })

    with open(path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("(no solution)\n")

    logger.info("Wrote %s  (%d rows)", path, len(rows))
    return path


def solution_fully_feasible(solution: dict) -> bool:
    """True if every week's ``totals.feasible`` is True."""
    for week_data in solution.get("weeks", []):
        if not week_data.get("totals", {}).get("feasible", False):
            return False
    return bool(solution.get("weeks"))


def print_solution_summary(solution: dict) -> None:
    """Print human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("SOLUTION SUMMARY")
    print("=" * 70)
    grand = 0.0
    for week_data in solution.get("weeks", []):
        t = week_data["totals"]
        print(f"\nWeek: {week_data['week']}")
        print(f"  Orders total     : {t['n_dedicated_orders'] + t['n_tendered_orders']}")
        print(f"  Dedicated orders : {t['n_dedicated_orders']}  ({t['n_active_trucks']} trucks active)")
        print(f"  Tendered orders  : {t['n_tendered_orders']}")
        print(f"  Dedicated cost   : ${t['dedicated_cost']:>12,.2f}")
        print(f"  Tender cost      : ${t['tender_cost']:>12,.2f}")
        print(f"  Week total       : ${t['grand_total']:>12,.2f}")
        print(f"  Feasible         : {t['feasible']}")
        if t["constraint_violations"]:
            print(f"  Violations       :")
            for v in t["constraint_violations"]:
                for viol in v["violations"]:
                    print(f"    [{v['truck']}] {viol['constraint']}: {viol['detail']}")
        grand += t["grand_total"]

    print(f"\n{'-'*70}")
    print(f"  GRAND TOTAL (all weeks): ${grand:>12,.2f}")
    print("=" * 70 + "\n")

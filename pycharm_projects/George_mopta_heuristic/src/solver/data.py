"""
data.py — Load and validate the competition Excel workbook into typed objects.

Sheet layout (confirmed from AIMMS-MOPTA Interfor data.xlsx):
  Locations    : Location (str), Latitude (float), Longitude (float)
  Shipments    : Order Number (int), Origin (str), Destination (str), Due Date (datetime)
  Carriers     : Carrier Home (str), Carrier (str)
  Lanes        : Origin (str), Destination (str), Mileage (float), Cost (float)
  Difficult Lanes: Origin (str), Destination (str), Additional Cost $/Mile (float)

All location codes are "CITY,ST" format (e.g., "ALBANY,GA").
Difficult Lanes destinations may be state codes ("TX") or ZIP-prefix codes ("GA_303").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Domain objects
# ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Order:
    order_id: int
    origin: str
    destination: str
    due_date: pd.Timestamp
    week_label: str  # "2026-01-17" or "2026-01-24"
    order_idx: int   # 0-based row index in the full orders list


@dataclass(frozen=True)
class Truck:
    truck_id: int      # 0-based index
    name: str
    home: str          # "CITY,ST"
    home_lat: float
    home_lon: float


@dataclass(frozen=True)
class DifficultLane:
    origin: str             # exact city name, e.g. "ALBANY,GA"
    destination_key: str    # state code "TX" or ZIP-prefix "GA_303"
    adder_per_mile: float   # additional $/mile for open-market tenders


@dataclass
class ProblemData:
    """All parsed problem data, plus pre-computed lookup structures."""
    orders: List[Order]
    trucks: List[Truck]
    locations: Dict[str, Tuple[float, float]]   # "CITY,ST" -> (lat, lon)
    lane_cost: Dict[Tuple[str, str], float]     # (origin, dest) -> $ cost
    lane_mileage: Dict[Tuple[str, str], float]  # (origin, dest) -> miles
    difficult_lanes: List[DifficultLane]
    orders_by_week: Dict[str, List[Order]]      # week_label -> orders

    # Derived numpy arrays (populated by build_arrays())
    loc_names: List[str] = field(default_factory=list)          # sorted location names
    loc_idx: Dict[str, int] = field(default_factory=dict)       # name -> matrix index
    cost_matrix: Optional[np.ndarray] = field(default=None)     # (N,N) float64
    mileage_matrix: Optional[np.ndarray] = field(default=None)  # (N,N) float64

    def build_arrays(self) -> None:
        """Pre-compute numpy matrices for fast GA evaluation."""
        self.loc_names = sorted(self.locations.keys())
        self.loc_idx = {loc: i for i, loc in enumerate(self.loc_names)}
        n = len(self.loc_names)
        self.cost_matrix = np.full((n, n), np.nan, dtype=np.float64)
        self.mileage_matrix = np.full((n, n), np.nan, dtype=np.float64)

        missing = 0
        for (orig, dest), cost in self.lane_cost.items():
            i = self.loc_idx.get(orig)
            j = self.loc_idx.get(dest)
            if i is None or j is None:
                missing += 1
                continue
            self.cost_matrix[i, j] = cost
            self.mileage_matrix[i, j] = self.lane_mileage[(orig, dest)]

        if missing:
            logger.warning("build_arrays: %d lane pairs had unresolvable location names", missing)

        # Fill NaN on diagonal with 0 mileage / minimum cost as fallback
        diag_nan = np.isnan(np.diag(self.mileage_matrix))
        if diag_nan.any():
            for k in np.where(diag_nan)[0]:
                self.mileage_matrix[k, k] = 0.0
                self.cost_matrix[k, k] = 400.0  # observed minimum in data


# ──────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────

def _norm_city(name: str) -> str:
    """Normalize a city code to uppercase-stripped form, e.g. 'Jacksonville,FL' -> 'JACKSONVILLE,FL'."""
    return name.strip().upper()


def load_problem(
    xlsx_path: str | Path,
    cfg_sheets: Optional[dict] = None,
) -> ProblemData:
    """
    Parse the competition Excel workbook and return a fully populated ProblemData.

    Parameters
    ----------
    xlsx_path : path to 'AIMMS-MOPTA Interfor data.xlsx'
    cfg_sheets : optional dict overriding sheet names (keys: locations, shipments,
                 carriers, lanes, difficult_lanes)
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Data file not found: {xlsx_path}")

    sheets = {
        "locations": "Locations",
        "shipments": "Shipments",
        "carriers": "Carriers",
        "lanes": "Lanes",
        "difficult_lanes": "Difficult Lanes",
    }
    if cfg_sheets:
        sheets.update(cfg_sheets)

    logger.info("Loading workbook: %s", xlsx_path)
    raw: dict[str, pd.DataFrame] = pd.read_excel(xlsx_path, sheet_name=None)

    # ── Locations ────────────────────────────────────────────────
    loc_df = raw[sheets["locations"]]
    _require_cols(loc_df, ["Location", "Latitude", "Longitude"], "Locations")
    locations: Dict[str, Tuple[float, float]] = {
        _norm_city(row["Location"]): (float(row["Latitude"]), float(row["Longitude"]))
        for _, row in loc_df.iterrows()
    }
    logger.info("Loaded %d locations", len(locations))

    # ── Carriers ─────────────────────────────────────────────────
    car_df = raw[sheets["carriers"]]
    _require_cols(car_df, ["Carrier Home", "Carrier"], "Carriers")
    trucks: List[Truck] = []
    for truck_id, (_, row) in enumerate(car_df.iterrows()):
        home = _norm_city(str(row["Carrier Home"]))
        if home not in locations:
            logger.warning("Carrier home '%s' not in Locations table", home)
            lat, lon = 0.0, 0.0
        else:
            lat, lon = locations[home]
        trucks.append(Truck(
            truck_id=truck_id,
            name=str(row["Carrier"]),
            home=home,
            home_lat=lat,
            home_lon=lon,
        ))
    logger.info("Loaded %d trucks", len(trucks))

    # ── Shipments ────────────────────────────────────────────────
    ship_df = raw[sheets["shipments"]]
    _require_cols(ship_df, ["Order Number", "Origin", "Destination", "Due Date"], "Shipments")
    orders: List[Order] = []
    unknown_locs: set[str] = set()
    for idx, (_, row) in enumerate(ship_df.iterrows()):
        origin = _norm_city(str(row["Origin"]))
        dest = _norm_city(str(row["Destination"]))
        due = pd.Timestamp(row["Due Date"])
        week_label = due.strftime("%Y-%m-%d")
        for loc in (origin, dest):
            if loc not in locations:
                unknown_locs.add(loc)
        orders.append(Order(
            order_id=int(row["Order Number"]),
            origin=origin,
            destination=dest,
            due_date=due,
            week_label=week_label,
            order_idx=idx,
        ))
    if unknown_locs:
        logger.warning("Shipment locations not in Locations table: %s", unknown_locs)
    logger.info("Loaded %d orders across due dates: %s",
                len(orders),
                sorted({o.week_label for o in orders}))

    orders_by_week: Dict[str, List[Order]] = {}
    for o in orders:
        orders_by_week.setdefault(o.week_label, []).append(o)

    # ── Lanes ────────────────────────────────────────────────────
    lane_df = raw[sheets["lanes"]]
    _require_cols(lane_df, ["Origin", "Destination", "Mileage", "Cost"], "Lanes")
    lane_cost: Dict[Tuple[str, str], float] = {}
    lane_mileage: Dict[Tuple[str, str], float] = {}
    for _, row in lane_df.iterrows():
        key = (_norm_city(str(row["Origin"])), _norm_city(str(row["Destination"])))
        lane_cost[key] = float(row["Cost"])
        lane_mileage[key] = float(row["Mileage"])
    logger.info("Loaded %d lane pairs", len(lane_cost))

    # Validate all order O/D pairs have a lane entry
    missing_lanes = 0
    for o in orders:
        if (o.origin, o.destination) not in lane_cost:
            logger.warning("No lane for order %d: %s -> %s", o.order_id, o.origin, o.destination)
            missing_lanes += 1
    if missing_lanes == 0:
        logger.info("All order O/D pairs have lane entries.")
    else:
        logger.warning("%d orders missing lane entries (will use 0 cost fallback)", missing_lanes)

    # ── Difficult Lanes ──────────────────────────────────────────
    diff_df = raw[sheets["difficult_lanes"]]
    _require_cols(diff_df, ["Origin", "Destination", "Additional Cost $/Mile"], "Difficult Lanes")
    difficult_lanes: List[DifficultLane] = []
    for _, row in diff_df.iterrows():
        # Origin is a city code (normalize); destination_key is a state/ZIP pattern (keep as-is)
        difficult_lanes.append(DifficultLane(
            origin=_norm_city(str(row["Origin"])),
            destination_key=str(row["Destination"]).strip(),
            adder_per_mile=float(row["Additional Cost $/Mile"]),
        ))
    logger.info("Loaded %d difficult lane records", len(difficult_lanes))

    problem = ProblemData(
        orders=orders,
        trucks=trucks,
        locations=locations,
        lane_cost=lane_cost,
        lane_mileage=lane_mileage,
        difficult_lanes=difficult_lanes,
        orders_by_week=orders_by_week,
    )
    problem.build_arrays()
    return problem


# ──────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────

def _require_cols(df: pd.DataFrame, cols: list[str], sheet: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet '{sheet}' missing columns: {missing}. Found: {list(df.columns)}")


def print_validation_report(problem: ProblemData) -> None:
    """Print a human-readable data quality summary."""
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"Locations  : {len(problem.locations)}")
    print(f"Trucks     : {len(problem.trucks)}")
    print(f"Orders     : {len(problem.orders)}")
    for wk, ords in sorted(problem.orders_by_week.items()):
        print(f"  Week {wk} : {len(ords)} orders")
    print(f"Lane pairs : {len(problem.lane_cost)}")
    print(f"Diff. lanes: {len(problem.difficult_lanes)}")

    # Cost range
    costs = list(problem.lane_cost.values())
    print(f"Lane cost  : min=${min(costs):.2f}  max=${max(costs):.2f}  mean=${sum(costs)/len(costs):.2f}")

    miles = list(problem.lane_mileage.values())
    print(f"Lane miles : min={min(miles):.1f}  max={max(miles):.1f}  mean={sum(miles)/len(miles):.1f}")

    # Truck homes in locations?
    bad_homes = [t for t in problem.trucks if t.home not in problem.locations]
    if bad_homes:
        print(f"WARNING: {len(bad_homes)} truck homes not in Locations: {[t.home for t in bad_homes]}")
    else:
        print("All truck homes are in Locations table. OK")

    # Matrix NaN check
    if problem.cost_matrix is not None:
        nan_count = int(np.isnan(problem.cost_matrix).sum())
        print(f"Cost matrix NaNs: {nan_count} / {problem.cost_matrix.size}")
    print("=" * 60)

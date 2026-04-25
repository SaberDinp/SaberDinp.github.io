"""
education.py — Intra-route local-search (LS) operators for memetic GA education.

Three first-improvement operators, applied per truck:
  or_opt_1  : relocate a single order to the best available position
  two_opt   : reverse a contiguous sub-sequence [i..j]
  or_opt_2  : relocate a consecutive pair to the best available position

LS objective: minimise  total_deadhead_miles
                        + home_penalty * max(0, final_return_dist - 100)

The assignment (which orders go to which truck) is NOT changed here.
Only the *route order within* a truck is improved.

Route-key encoding
------------------
Keys are float32 arrays parallel to the chromosome (one value per local order
position).  For truck t, its assigned orders are sorted by keys to determine
route order.  After LS, keys are updated to encode the improved permutation so
the improvement survives into the next generation's crossover.

  keys[local_pos] ∈ [0, 1)   — lower key → earlier in route

Usage in ga.py
--------------
  keys = init_keys_from_nn(chrom, evaluator)   # start from NN route
  chrom, keys = educate_individual(chrom, keys, evaluator, cfg_edu)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .data import Order, Truck, ProblemData

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# LS objective
# ──────────────────────────────────────────────────────────────────

def _route_obj(
    route: List[Order],
    truck: Truck,
    mm: np.ndarray,
    loc_idx: dict,
    home_penalty: float,
    max_home_dist: float,
) -> float:
    """
    Compute LS objective for a given route.

    = total deadhead miles (reposition + return home)
    + home_penalty * max(0, return_dist_from_home - max_home_dist)
    """
    if not route:
        return 0.0

    home_i = loc_idx.get(truck.home, 0)
    orig_idx = np.array([loc_idx.get(o.origin, 0) for o in route], dtype=np.intp)
    dest_idx = np.array([loc_idx.get(o.destination, 0) for o in route], dtype=np.intp)
    n = len(route)

    # from_arr[k] -> to_arr[k] are the deadhead hops:
    #   0: home -> route[0].origin
    #   k: route[k-1].dest -> route[k].origin   (k = 1..n-1)
    #   n: route[n-1].dest -> home  (return)
    from_arr = np.empty(n + 1, dtype=np.intp)
    to_arr = np.empty(n + 1, dtype=np.intp)
    from_arr[0] = home_i
    from_arr[1:n] = dest_idx[:-1]
    from_arr[n] = dest_idx[-1]
    to_arr[:n] = orig_idx
    to_arr[n] = home_i

    dh = float(np.sum(mm[from_arr, to_arr]))
    final_dist = float(mm[dest_idx[-1], home_i])
    dh += home_penalty * max(0.0, final_dist - max_home_dist)
    return dh


# ──────────────────────────────────────────────────────────────────
# LS operators (first-improvement)
# ──────────────────────────────────────────────────────────────────

def or_opt_1(
    route: List[Order],
    truck: Truck,
    mm: np.ndarray,
    loc_idx: dict,
    home_penalty: float = 0.0,
    max_home_dist: float = 100.0,
) -> List[Order]:
    """
    Or-opt-1: for each order at position i, try inserting it at every other
    position j.  Accept the first move that strictly reduces the LS objective.

    Repeats until no improving move exists.
    """
    n = len(route)
    if n < 2:
        return route

    best = list(route)
    best_cost = _route_obj(best, truck, mm, loc_idx, home_penalty, max_home_dist)

    improved = True
    while improved:
        improved = False
        for i in range(n):
            order = best[i]
            rest = best[:i] + best[i + 1:]
            for j in range(len(rest) + 1):
                if j == i:
                    continue
                candidate = rest[:j] + [order] + rest[j:]
                cost = _route_obj(candidate, truck, mm, loc_idx, home_penalty, max_home_dist)
                if cost < best_cost - 1e-6:
                    best = candidate
                    best_cost = cost
                    improved = True
                    break
            if improved:
                break

    return best


def two_opt(
    route: List[Order],
    truck: Truck,
    mm: np.ndarray,
    loc_idx: dict,
    home_penalty: float = 0.0,
    max_home_dist: float = 100.0,
) -> List[Order]:
    """
    2-opt: try reversing every sub-sequence [i..j].
    Accept the first move that strictly reduces the LS objective.
    Repeats until no improving move exists.
    """
    n = len(route)
    if n < 3:
        return route

    best = list(route)
    best_cost = _route_obj(best, truck, mm, loc_idx, home_penalty, max_home_dist)

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                candidate = best[:i] + best[i:j][::-1] + best[j:]
                cost = _route_obj(candidate, truck, mm, loc_idx, home_penalty, max_home_dist)
                if cost < best_cost - 1e-6:
                    best = candidate
                    best_cost = cost
                    improved = True
                    break
            if improved:
                break

    return best


def or_opt_2(
    route: List[Order],
    truck: Truck,
    mm: np.ndarray,
    loc_idx: dict,
    home_penalty: float = 0.0,
    max_home_dist: float = 100.0,
) -> List[Order]:
    """
    Or-opt-2: try relocating each consecutive pair to every other position.
    Accept the first improving move.  Repeats until no improvement.
    """
    n = len(route)
    if n < 3:
        return route

    best = list(route)
    best_cost = _route_obj(best, truck, mm, loc_idx, home_penalty, max_home_dist)

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            pair = best[i:i + 2]
            rest = best[:i] + best[i + 2:]
            for j in range(len(rest) + 1):
                candidate = rest[:j] + pair + rest[j:]
                cost = _route_obj(candidate, truck, mm, loc_idx, home_penalty, max_home_dist)
                if cost < best_cost - 1e-6:
                    best = candidate
                    best_cost = cost
                    improved = True
                    break
            if improved:
                break

    return best


# ──────────────────────────────────────────────────────────────────
# Route-level education
# ──────────────────────────────────────────────────────────────────

def educate_route(
    route: List[Order],
    truck: Truck,
    problem: ProblemData,
    cfg_edu: dict,
) -> List[Order]:
    """
    Apply enabled LS operators to *route* for up to ``max_ls_iters`` passes.
    Returns the improved (or unchanged) route.

    cfg_edu keys:
      home_penalty_per_mile  (float, default 10.0)
      max_home_dist          (float, default 100.0)
      max_ls_iters           (int, default 3)
      or_opt_1               (bool, default True)
      two_opt                (bool, default True)
      or_opt_2               (bool, default True)
    """
    if len(route) < 2:
        return route

    mm = problem.mileage_matrix
    loc_idx = problem.loc_idx
    hp = cfg_edu.get("home_penalty_per_mile", 10.0)
    max_hd = cfg_edu.get("max_home_dist", 100.0)
    max_iters = cfg_edu.get("max_ls_iters", 3)
    kw = dict(truck=truck, mm=mm, loc_idx=loc_idx,
              home_penalty=hp, max_home_dist=max_hd)

    for _ in range(max_iters):
        prev = list(route)
        if cfg_edu.get("or_opt_1", True):
            route = or_opt_1(route, **kw)
        if cfg_edu.get("two_opt", True):
            route = two_opt(route, **kw)
        if cfg_edu.get("or_opt_2", True):
            route = or_opt_2(route, **kw)
        if route == prev:
            break

    return route


# ──────────────────────────────────────────────────────────────────
# Route-key helpers
# ──────────────────────────────────────────────────────────────────

def init_keys_from_nn(
    chrom: np.ndarray,
    evaluator,  # FitnessEvaluator — imported lazily to avoid circular import
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Return a keys array (float32, shape n_orders) whose values encode the
    NN-derived route order for each truck.

    Orders not assigned to any truck get a random key in [0,1).
    """
    from .scheduler import build_nearest_neighbor_route  # local import OK

    n = evaluator.n_orders
    problem = evaluator.problem
    rng = rng if rng is not None else np.random.default_rng(0)
    keys = rng.random(n).astype(np.float32)

    for t_id in range(evaluator.n_trucks):
        local_positions = np.where(chrom == t_id)[0]
        if len(local_positions) == 0:
            continue
        truck = problem.trucks[t_id]
        global_indices = evaluator.week_order_indices[local_positions]
        nn_route = build_nearest_neighbor_route(truck, global_indices, problem)
        n_truck = len(local_positions)
        oid_to_lp = {
            problem.orders[gi].order_id: lp
            for gi, lp in zip(global_indices, local_positions)
        }
        for rank, order in enumerate(nn_route):
            lp = oid_to_lp.get(order.order_id)
            if lp is not None:
                keys[lp] = float(rank) / max(n_truck - 1, 1)

    return keys


def _update_keys_from_route(
    improved_route: List[Order],
    local_positions: np.ndarray,
    global_indices: np.ndarray,
    problem: ProblemData,
    keys: np.ndarray,
) -> None:
    """In-place: update keys[local_pos] to encode the rank in improved_route."""
    n_truck = len(local_positions)
    oid_to_lp = {
        problem.orders[gi].order_id: lp
        for gi, lp in zip(global_indices, local_positions)
    }
    for rank, order in enumerate(improved_route):
        lp = oid_to_lp.get(order.order_id)
        if lp is not None:
            keys[lp] = float(rank) / max(n_truck - 1, 1)


# ──────────────────────────────────────────────────────────────────
# Individual-level education (called from ga.py)
# ──────────────────────────────────────────────────────────────────

def educate_individual(
    chrom: np.ndarray,
    keys: np.ndarray,
    evaluator,  # FitnessEvaluator
    cfg_edu: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply intra-route LS to every truck assigned orders in *chrom*.
    Updates *keys* to reflect the LS-improved route order.

    Parameters
    ----------
    chrom   : int8 array (n_orders,) — assignment chromosome (unchanged)
    keys    : float32 array (n_orders,) — route-key encoding (updated in-place copy)
    evaluator : FitnessEvaluator instance
    cfg_edu : education sub-config dict

    Returns
    -------
    (chrom, new_keys) — chrom is unchanged; new_keys encodes improved routes
    """
    problem = evaluator.problem
    new_keys = keys.copy()

    for t_id in range(evaluator.n_trucks):
        local_positions = np.where(chrom == t_id)[0]
        if len(local_positions) < 2:
            continue

        truck = problem.trucks[t_id]
        global_indices = evaluator.week_order_indices[local_positions]

        # Reconstruct route from current keys
        key_vals = keys[local_positions]
        sorted_order = np.argsort(key_vals)          # indices into local_positions
        route_global = global_indices[sorted_order]
        route = [problem.orders[i] for i in route_global]

        # Apply LS
        old_seq = tuple(o.order_id for o in route)
        improved = educate_route(route, truck, problem, cfg_edu)
        new_seq = tuple(o.order_id for o in improved)
        if new_seq != old_seq:
            _update_keys_from_route(
                improved, local_positions, global_indices, problem, new_keys
            )

    return chrom, new_keys

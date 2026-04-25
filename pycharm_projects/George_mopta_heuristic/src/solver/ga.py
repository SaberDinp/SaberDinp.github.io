"""
ga.py — Genetic Algorithm solver for the Interfor truck routing problem.

Method choice: Memetic GA (GA + repair + local-search perturbation)
  - Plain GA: fast but poor on VRP-style problems with hard constraints.
  - Permutation GA (DEAP/pymoo): powerful but heavyweight dependency.
  - Our choice: hand-rolled GA on a flat integer assignment chromosome,
    with a nearest-neighbor route heuristic and a repair operator.
    This keeps dependencies minimal (numpy only) and is easy to tune.

Chromosome encoding (Layer A only — routes derived, not evolved):
  chrom[i] ∈ {-1, 0 .. n_trucks-1} for i = 0..n_orders-1
    -1      : order i is tendered to open market
    0..19   : order i is assigned to truck id k

Route order: for each truck, derived once per fitness evaluation via a
  nearest-neighbor heuristic from home location.

Fitness: total freight cost (dedicated + tender) + soft penalties.
  See cost.py and feasibility.py for details.

Operators:
  Initialization : three seeds + random population
    1. "all-tender" baseline (all orders tendered)
    2. Greedy dedicated: assign orders on difficult lanes to trucks near origin
    3. Balanced random: uniform assignment to trucks or tender
  Selection      : tournament (size configurable)
  Crossover      : uniform crossover on assignment vector
  Mutation       : gene-wise: with prob p, reassign to random truck or tender
  Elitism        : top-k individuals carried forward unchanged
  Termination    : max generations OR stagnation OR wall-clock cap

Two-week handling: the GA is run independently on each week's orders.
  Within each run, the chromosome covers only that week's orders.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from typing import List, Tuple, Optional, Dict

import numpy as np

from .data import ProblemData, Order, Truck
from .cost import (
    build_order_base_cost_array,
    build_tender_surcharge_array,
    total_solution_cost,
    truck_dedicated_cost,
)
from .feasibility import soft_penalty, repair_truck_route
from .scheduler import build_nearest_neighbor_route, schedule_truck_week
from .education import init_keys_from_nn, educate_individual

logger = logging.getLogger(__name__)


_WORKER_CONTEXT: Dict[str, object] = {}


def _init_eval_worker(
    problem: ProblemData,
    week_order_indices: np.ndarray,
    cfg_constraints: dict,
    cfg_penalties: dict,
) -> None:
    """Cache evaluator state once per spawned worker process."""
    _WORKER_CONTEXT["evaluator"] = FitnessEvaluator(
        problem=problem,
        week_order_indices=week_order_indices,
        cfg_constraints=cfg_constraints,
        cfg_penalties=cfg_penalties,
    )


def _evaluate_worker(task: Tuple[np.ndarray, Optional[np.ndarray]]) -> float:
    chrom, keys = task
    evaluator = _WORKER_CONTEXT["evaluator"]
    return float(evaluator.evaluate(chrom, keys))


def _educate_worker(task: Tuple[np.ndarray, np.ndarray, dict, int]) -> np.ndarray:
    chrom, keys, cfg_edu, seed = task
    evaluator = _WORKER_CONTEXT["evaluator"]
    rng = np.random.default_rng(seed)
    keyed = init_keys_from_nn(chrom, evaluator, rng=rng)
    _, new_keys = educate_individual(chrom, keyed if keys is None else keys, evaluator, cfg_edu)
    return new_keys.astype(np.float32, copy=False)


def _resolve_worker_count(requested: Optional[int], task_count: int) -> int:
    """Normalise worker counts: <=1 disables parallelism, 0 means auto."""
    if task_count <= 1:
        return 1
    if requested is None:
        return 1
    if requested == 0:
        cpu_count = os.cpu_count() or 1
        requested = min(12, cpu_count)
    return max(1, min(int(requested), task_count))


def _resolve_elite_size(cfg_ga: Dict, pop_size: int) -> int:
    """
    Elite count for slicing / loops: always an int in ``[1, pop_size - 1]``.

    If ``elite_fraction`` is set to a number in ``(0, 1]``, it overrides
    ``elite_size`` and uses ``round(pop_size * fraction)``. Values outside
    ``(0, 1]`` are ignored and ``elite_size`` is used (so typos do not wipe
    the population). Internally the GA always uses a whole elite count; the
    fraction is only a convenience when scaling ``population_size``.
    """
    frac = cfg_ga.get("elite_fraction")
    if frac is not None:
        f = float(frac)
        if 0.0 < f <= 1.0:
            e = int(round(pop_size * f))
        else:
            e = int(cfg_ga.get("elite_size", 4))
    else:
        e = int(cfg_ga.get("elite_size", 4))
    return max(1, min(e, pop_size - 1))


def _should_parallelize(task_count: int, n_orders: int, workers: int) -> bool:
    """Avoid process-spawn overhead on tiny batches."""
    return workers > 1 and task_count >= workers * 2 and (task_count * n_orders) >= 2000


# ──────────────────────────────────────────────────────────────────
# Fitness evaluator (vectorized where possible)
# ──────────────────────────────────────────────────────────────────

class FitnessEvaluator:
    """
    Pre-caches arrays for fast chromosome fitness evaluation.
    One instance per (problem, week_order_indices) pair.
    """

    def __init__(
        self,
        problem: ProblemData,
        week_order_indices: np.ndarray,   # row indices into problem.orders
        cfg_constraints: dict,
        cfg_penalties: dict,
    ):
        self.problem = problem
        self.week_order_indices = week_order_indices
        self.n_orders = len(week_order_indices)
        self.n_trucks = len(problem.trucks)
        self.cfg_c = cfg_constraints
        self.cfg_p = cfg_penalties
        self.min_earnings = cfg_constraints.get("min_weekly_earnings", 4000.0)

        # Slice base costs and surcharges to this week's orders
        full_base = build_order_base_cost_array(problem)
        full_surcharge = build_tender_surcharge_array(problem)
        self.base_cost = full_base[week_order_indices]      # (n_orders,)  local-indexed
        self.surcharge = full_surcharge[week_order_indices]  # (n_orders,)  local-indexed
        self.full_base_cost = full_base                      # (N_total,)   global-indexed

        # Map local order position -> problem-global order object
        self.local_orders = [problem.orders[i] for i in week_order_indices]

    def evaluate(self, chrom: np.ndarray, keys: Optional[np.ndarray] = None) -> float:
        """
        Full fitness for a chromosome of length n_orders.

        Parameters
        ----------
        chrom : int8 array (n_orders,) — assignment chromosome
        keys  : float32 array (n_orders,) or None.
                If provided, orders for each truck are sorted by keys to
                determine route order (LS-key encoding).
                If None, a nearest-neighbor heuristic is used (Phase 1 behaviour).

        Returns scalar cost (lower = better).
        """
        assert len(chrom) == self.n_orders

        # ── Tender cost ──────────────────────────────────────────
        tender_mask = chrom == -1
        tender_cost = float(np.sum(
            self.base_cost[tender_mask] + self.surcharge[tender_mask]
        ))

        # ── Dedicated cost + penalties ───────────────────────────
        dedicated_cost = 0.0
        penalty = 0.0
        problem = self.problem

        for t_id in range(self.n_trucks):
            local_positions = np.where(chrom == t_id)[0]
            if len(local_positions) == 0:
                continue
            truck = problem.trucks[t_id]

            # Convert local chromosome positions -> global problem.orders indices
            global_indices = self.week_order_indices[local_positions]

            if keys is not None:
                # Route order from key-sorted positions (LS-improved encoding)
                key_vals = keys[local_positions]
                sorted_order = np.argsort(key_vals)
                route_global = global_indices[sorted_order]
                route = [problem.orders[i] for i in route_global]
            else:
                # Nearest-neighbor baseline
                route = build_nearest_neighbor_route(truck, global_indices, problem)

            sched = schedule_truck_week(
                truck, route, problem,
                self.cfg_c.get("max_miles_per_day", 450.0)
            )

            # Lane cost → guaranteed minimum (indexed by local positions)
            earnings = float(np.sum(self.base_cost[local_positions]))
            dedicated_cost += max(earnings, self.min_earnings)

            # Soft penalties for constraint violations
            penalty += soft_penalty(
                sched, truck, problem, self.cfg_c, self.cfg_p
            )

        return tender_cost + dedicated_cost + penalty

    def batch_evaluate(
        self,
        population: np.ndarray,
        pop_keys: Optional[np.ndarray] = None,
        max_workers: int = 1,
        executor: concurrent.futures.ProcessPoolExecutor | None = None,
    ) -> np.ndarray:
        """
        Evaluate all individuals.

        population shape : (pop_size, n_orders)
        pop_keys shape   : (pop_size, n_orders) float32 or None
        """
        pop_size = len(population)
        if (
            executor is None
            or not _should_parallelize(pop_size, self.n_orders, max_workers)
        ):
            if pop_keys is None:
                return np.array([self.evaluate(population[i]) for i in range(pop_size)])
            return np.array([
                self.evaluate(population[i], pop_keys[i]) for i in range(pop_size)
            ])

        tasks = (
            [(population[i], None) for i in range(pop_size)]
            if pop_keys is None
            else [(population[i], pop_keys[i]) for i in range(pop_size)]
        )
        return np.fromiter(executor.map(_evaluate_worker, tasks), dtype=np.float64, count=pop_size)


# ──────────────────────────────────────────────────────────────────
# Population initialisation
# ──────────────────────────────────────────────────────────────────

def _init_all_tender(n_orders: int) -> np.ndarray:
    return np.full(n_orders, -1, dtype=np.int8)


def _init_balanced_random(
    n_orders: int,
    n_trucks: int,
    tender_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign each order randomly to a truck or tender."""
    chrom = rng.integers(0, n_trucks, size=n_orders, dtype=np.int8)
    tender_mask = rng.random(n_orders) < tender_prob
    chrom[tender_mask] = -1
    return chrom


def _init_greedy_difficult(
    evaluator: FitnessEvaluator,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Heuristic seed: assign orders with a difficult-lane surcharge to trucks,
    preferring trucks whose home is nearest to the order origin.
    Orders with no surcharge are tendered.
    """
    problem = evaluator.problem
    n_orders = evaluator.n_orders
    chrom = np.full(n_orders, -1, dtype=np.int8)
    mileage_matrix = problem.mileage_matrix
    loc_idx = problem.loc_idx

    truck_home_idx = np.array([loc_idx.get(t.home, 0) for t in problem.trucks])

    for local_i, order in enumerate(evaluator.local_orders):
        if evaluator.surcharge[local_i] <= 0.0:
            continue  # no difficult-lane benefit; tender
        orig_idx = loc_idx.get(order.origin, None)
        if orig_idx is None:
            continue
        # Find nearest truck home to order origin
        dists = mileage_matrix[truck_home_idx, orig_idx]
        nearest_truck = int(np.argmin(dists))
        chrom[local_i] = nearest_truck

    return chrom


def initialize_population(
    evaluator: FitnessEvaluator,
    pop_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return population array (pop_size, n_orders) seeded with heuristics."""
    n = evaluator.n_orders
    nt = evaluator.n_trucks
    population = np.zeros((pop_size, n), dtype=np.int8)

    # Seed 0: all tender
    population[0] = _init_all_tender(n)
    # Seed 1: difficult-lane greedy
    population[1] = _init_greedy_difficult(evaluator, rng)
    # Seed 2: 40% tender, rest random trucks
    population[2] = _init_balanced_random(n, nt, 0.40, rng)

    # Rest: random with varying tender probability
    for i in range(3, pop_size):
        tender_prob = rng.uniform(0.2, 0.7)
        population[i] = _init_balanced_random(n, nt, tender_prob, rng)

    return population


# ──────────────────────────────────────────────────────────────────
# Genetic operators
# ──────────────────────────────────────────────────────────────────

def tournament_select(
    fitness: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> int:
    """Return index of tournament winner (lowest fitness)."""
    candidates = rng.integers(0, len(fitness), size=tournament_size)
    return int(candidates[np.argmin(fitness[candidates])])


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover: each gene independently from either parent."""
    mask = rng.random(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2).astype(np.int8)
    child2 = np.where(mask, parent2, parent1).astype(np.int8)
    return child1, child2


def uniform_crossover_with_keys(
    parent1: np.ndarray,
    keys1: np.ndarray,
    parent2: np.ndarray,
    keys2: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Uniform crossover on both the chromosome and the route-key array.
    Each gene (and its key) is inherited from the same parent.

    Returns (child1, child1_keys, child2, child2_keys).
    """
    mask = rng.random(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2).astype(np.int8)
    child2 = np.where(mask, parent2, parent1).astype(np.int8)
    ck1 = np.where(mask, keys1, keys2).astype(np.float32)
    ck2 = np.where(mask, keys2, keys1).astype(np.float32)
    return child1, ck1, child2, ck2


def mutate(
    chrom: np.ndarray,
    n_trucks: int,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Gene-wise mutation: with probability mutation_rate, reassign each order
    to a random truck (0..n_trucks-1) or tender (-1).
    """
    chrom = chrom.copy()
    mask = rng.random(len(chrom)) < mutation_rate
    new_vals = rng.integers(0, n_trucks + 1, size=mask.sum(), dtype=np.int8) - 1
    chrom[mask] = new_vals
    return chrom


def apply_repair(
    chrom: np.ndarray,
    evaluator: FitnessEvaluator,
) -> np.ndarray:
    """
    Repair operator: for each truck, eject orders that push total miles beyond
    the 5-day cap or deadhead fraction over 50%.
    Ejected orders are moved to tender.
    """
    problem = evaluator.problem
    cfg_c = evaluator.cfg_c
    chrom = chrom.copy()

    for t_id in range(evaluator.n_trucks):
        local_positions = list(np.where(chrom == t_id)[0])
        if not local_positions:
            continue
        truck = problem.trucks[t_id]
        # Convert local chromosome positions -> global indices for repair
        global_indices = list(evaluator.week_order_indices[local_positions])
        # repair_truck_route works with global indices + full_base_cost
        kept_global, ejected_global = repair_truck_route(
            truck, global_indices, problem, evaluator.full_base_cost, cfg_c
        )
        ejected_set = set(ejected_global)
        # Map ejected global indices back to local positions and tender them
        for lp, gi in zip(local_positions, global_indices):
            if gi in ejected_set:
                chrom[lp] = -1

    return chrom


# ──────────────────────────────────────────────────────────────────
# Main GA loop
# ──────────────────────────────────────────────────────────────────

def run_ga(
    evaluator: FitnessEvaluator,
    pop_size: int = 60,
    max_generations: int = 300,
    elite_size: int = 4,
    crossover_rate: float = 0.80,
    mutation_rate: float = 0.04,
    tournament_size: int = 4,
    stagnation_limit: int = 1000,
    max_wall_seconds: float = 3600.0,
    seed: int = 42,
    repair_every_n_gens: int = 10,
    eval_workers: int = 1,
    education_enabled: bool = False,
    p_edu: float = 0.30,
    cfg_education: Optional[Dict] = None,
    edu_workers: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Run the GA for one week's orders.

    Parameters
    ----------
    education_enabled : enable intra-route LS education each generation
    p_edu             : probability of educating each non-elite individual
    cfg_education     : education sub-config dict (see configs/default.yaml)

    Returns
    -------
    best_chrom      : best assignment chromosome found
    best_fitness    : corresponding fitness value
    history         : list of best fitness per generation
    """
    rng = np.random.default_rng(seed)
    n_orders = evaluator.n_orders
    n_trucks = evaluator.n_trucks
    cfg_edu = cfg_education or {}
    edu_every = max(1, int(cfg_edu.get("every_n_generations", 1)))
    eval_workers = _resolve_worker_count(eval_workers, pop_size)
    edu_workers = _resolve_worker_count(edu_workers, max(0, pop_size - elite_size))
    eval_executor: concurrent.futures.ProcessPoolExecutor | None = None
    edu_executor: concurrent.futures.ProcessPoolExecutor | None = None

    if _should_parallelize(pop_size, evaluator.n_orders, eval_workers):
        eval_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=eval_workers,
            initializer=_init_eval_worker,
            initargs=(
                evaluator.problem,
                evaluator.week_order_indices,
                evaluator.cfg_c,
                evaluator.cfg_p,
            ),
        )

    if education_enabled and _should_parallelize(pop_size - elite_size, evaluator.n_orders, edu_workers):
        edu_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=edu_workers,
            initializer=_init_eval_worker,
            initargs=(
                evaluator.problem,
                evaluator.week_order_indices,
                evaluator.cfg_c,
                evaluator.cfg_p,
            ),
        )

    try:
        # ── Initialise ───────────────────────────────────────────────
        population = initialize_population(evaluator, pop_size, rng)

        # Route keys: each row encodes the LS-improved route order for one individual
        # Initialised from NN routes so generation-0 keys are already good
        pop_keys = np.zeros((pop_size, n_orders), dtype=np.float32)
        if education_enabled:
            for i in range(pop_size):
                pop_keys[i] = init_keys_from_nn(population[i], evaluator, rng=rng)

        fitness = evaluator.batch_evaluate(
            population,
            pop_keys if education_enabled else None,
            max_workers=eval_workers,
            executor=eval_executor,
        )
        best_idx = int(np.argmin(fitness))
        best_chrom = population[best_idx].copy()
        best_keys = pop_keys[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        history: List[float] = [best_fitness]

        stagnation = 0
        t0 = time.time()

        if verbose:
            logger.info(
                "GA start | orders=%d trucks=%d pop=%d gens=%d education=%s | eval_workers=%d edu_workers=%d | init best=%.2f",
                n_orders, n_trucks, pop_size, max_generations,
                "on" if education_enabled else "off", eval_workers, edu_workers, best_fitness,
            )

        # ── Evolution ────────────────────────────────────────────────
        for gen in range(1, max_generations + 1):
            if time.time() - t0 > max_wall_seconds:
                logger.info("Wall-clock cap reached at generation %d", gen)
                break

            # Elite: carry forward best k individuals (and their keys)
            elite_order = np.argsort(fitness)[:elite_size]
            new_pop = population[elite_order].copy()
            new_keys = pop_keys[elite_order].copy()

            # Fill rest via selection + crossover + mutation
            while len(new_pop) < pop_size:
                p1_idx = tournament_select(fitness, tournament_size, rng)
                p2_idx = tournament_select(fitness, tournament_size, rng)
                if rng.random() < crossover_rate:
                    if education_enabled:
                        c1, ck1, c2, ck2 = uniform_crossover_with_keys(
                            population[p1_idx], pop_keys[p1_idx],
                            population[p2_idx], pop_keys[p2_idx],
                            rng,
                        )
                    else:
                        c1, c2 = uniform_crossover(population[p1_idx], population[p2_idx], rng)
                        ck1 = ck2 = None
                else:
                    c1 = population[p1_idx].copy()
                    c2 = population[p2_idx].copy()
                    ck1 = pop_keys[p1_idx].copy() if education_enabled else None
                    ck2 = pop_keys[p2_idx].copy() if education_enabled else None

                c1 = mutate(c1, n_trucks, mutation_rate, rng)
                c2 = mutate(c2, n_trucks, mutation_rate, rng)

                if education_enabled:
                    # Mutation changes assignments; crossover-inherited keys are stale.
                    ck1 = init_keys_from_nn(c1, evaluator, rng=rng)
                    ck2 = init_keys_from_nn(c2, evaluator, rng=rng)
                    new_pop = np.vstack([new_pop, c1, c2])
                    new_keys = np.vstack([new_keys, ck1, ck2])
                else:
                    new_pop = np.vstack([new_pop, c1, c2])
                    new_keys = np.vstack([new_keys, np.zeros_like(new_keys[:1]), np.zeros_like(new_keys[:1])])

            population = new_pop[:pop_size]
            pop_keys = new_keys[:pop_size]

            # Periodic repair pass (skip elites)
            if gen % repair_every_n_gens == 0:
                for i in range(elite_size, pop_size):
                    population[i] = apply_repair(population[i], evaluator)
                    if education_enabled:
                        pop_keys[i] = init_keys_from_nn(population[i], evaluator, rng=rng)

            # Education pass: improve non-elite individuals' routes with LS
            if education_enabled and (gen % edu_every == 0):
                selected = [
                    i for i in range(elite_size, pop_size)
                    if rng.random() < p_edu
                ]
                if selected:
                    if edu_executor is None or not _should_parallelize(len(selected), evaluator.n_orders, edu_workers):
                        for i in selected:
                            _, pop_keys[i] = educate_individual(
                                population[i], pop_keys[i], evaluator, cfg_edu
                            )
                    else:
                        edu_tasks = []
                        for i in selected:
                            child_seed = int(
                                np.random.SeedSequence([seed, gen, i]).generate_state(1, dtype=np.uint64)[0]
                            )
                            edu_tasks.append((population[i], pop_keys[i], cfg_edu, child_seed))
                        for i, new_key in zip(selected, edu_executor.map(_educate_worker, edu_tasks)):
                            pop_keys[i] = new_key

            # Evaluate
            fitness = evaluator.batch_evaluate(
                population,
                pop_keys if education_enabled else None,
                max_workers=eval_workers,
                executor=eval_executor,
            )
            gen_best_idx = int(np.argmin(fitness))
            gen_best = float(fitness[gen_best_idx])

            if gen_best < best_fitness - 1e-6:
                best_fitness = gen_best
                best_chrom = population[gen_best_idx].copy()
                best_keys = pop_keys[gen_best_idx].copy()
                stagnation = 0
            else:
                stagnation += 1

            history.append(best_fitness)

            if verbose and gen % 10 == 0:
                logger.info(
                    "Gen %3d | best=%.2f | stagnation=%d | elapsed=%.1fs",
                    gen, best_fitness, stagnation, time.time() - t0,
                )

            if stagnation >= stagnation_limit:
                logger.info("Stagnation limit reached at generation %d", gen)
                break

        elapsed = time.time() - t0
        logger.info(
            "GA done | best fitness=%.2f | generations=%d | time=%.1fs",
            best_fitness, len(history), elapsed,
        )

        # Final feasibility repair (elites skip periodic repair during evolution)
        best_chrom = apply_repair(best_chrom, evaluator)
        if education_enabled:
            best_keys = init_keys_from_nn(best_chrom, evaluator, rng=rng)
            _, best_keys = educate_individual(best_chrom, best_keys, evaluator, cfg_edu)
            best_fitness = float(evaluator.evaluate(best_chrom, best_keys))
        else:
            best_fitness = float(evaluator.evaluate(best_chrom))

        return best_chrom, best_fitness, history
    finally:
        if eval_executor is not None:
            eval_executor.shutdown()
        if edu_executor is not None:
            edu_executor.shutdown()


# ──────────────────────────────────────────────────────────────────
# Multi-week driver
# ──────────────────────────────────────────────────────────────────

def solve_all_weeks(
    problem: ProblemData,
    cfg: dict,
    verbose: bool = True,
) -> Dict[str, Tuple]:
    """
    Run GA independently for each planning week.

    Returns dict: week_label -> (best_chrom_local, best_fitness, history, baseline_T, baseline_G)
    where best_chrom_local has length = number of orders in that week.
    """
    cfg_ga = cfg.get("ga", {})
    cfg_c = cfg.get("constraints", {})
    cfg_p = cfg.get("penalties", {})

    results: Dict[str, Tuple[np.ndarray, float, List[float]]] = {}

    for week_label, week_orders in sorted(problem.orders_by_week.items()):
        logger.info("=" * 60)
        logger.info("Solving week: %s  (%d orders)", week_label, len(week_orders))
        logger.info("=" * 60)

        week_order_indices = np.array([o.order_idx for o in week_orders])

        # Optionally limit to a subset for debugging
        subset = cfg_ga.get("order_subset", None)
        if subset is not None and subset < len(week_order_indices):
            logger.info("DEBUG: limiting to first %d orders", subset)
            week_order_indices = week_order_indices[:subset]

        evaluator = FitnessEvaluator(
            problem=problem,
            week_order_indices=week_order_indices,
            cfg_constraints=cfg_c,
            cfg_penalties=cfg_p,
        )

        cfg_edu = cfg.get("education", {})
        edu_enabled = cfg_edu.get("enabled", False)

        # ── Baselines ────────────────────────────────────────────
        tender_chrom = np.full(len(week_order_indices), -1, dtype=np.int8)
        baseline_T = float(evaluator.evaluate(tender_chrom))
        greedy_chrom = _init_greedy_difficult(evaluator, np.random.default_rng(0))
        baseline_G = float(evaluator.evaluate(greedy_chrom))
        logger.info(
            "Baselines | T (all-tender)=%.2f | G (greedy-difficult)=%.2f",
            baseline_T, baseline_G,
        )

        pop_size = int(cfg_ga.get("population_size", 60))
        best_chrom, best_fitness, history = run_ga(
            evaluator=evaluator,
            pop_size=pop_size,
            max_generations=cfg_ga.get("max_generations", 300),
            elite_size=_resolve_elite_size(cfg_ga, pop_size),
            crossover_rate=cfg_ga.get("crossover_rate", 0.80),
            mutation_rate=cfg_ga.get("mutation_rate_gene", 0.04),
            tournament_size=cfg_ga.get("tournament_size", 4),
            stagnation_limit=cfg_ga.get("stagnation_limit", 1000),
            max_wall_seconds=cfg_ga.get("max_wall_seconds", 3600.0),
            seed=cfg_ga.get("seed", 42),
            eval_workers=cfg_ga.get("eval_workers", 1),
            education_enabled=edu_enabled,
            p_edu=cfg_edu.get("p_edu", 0.30),
            cfg_education=cfg_edu,
            edu_workers=cfg_edu.get("workers", 1),
            verbose=verbose,
        )

        logger.info(
            "Baselines vs GA | T=%.2f | G=%.2f | C=%.2f (GA best)",
            baseline_T, baseline_G, best_fitness,
        )

        results[week_label] = (best_chrom, best_fitness, history, baseline_T, baseline_G)

    return results

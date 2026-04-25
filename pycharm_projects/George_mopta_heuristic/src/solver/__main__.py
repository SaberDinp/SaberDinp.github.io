"""
__main__.py — CLI entry point.

Usage:
  python -m solver --data "AIMMS_MOPTA_2026/AIMMS-MOPTA Interfor data.xlsx"
  python -m solver --data PATH --seed 0 --generations 50 --pop 30 --mutation 0.06 --elite-fraction 0.004
  python -m solver --data PATH --subset 50     # debug on 50 orders
  python -m solver --data PATH --validate-only # data check, no GA

All options override the corresponding entry in configs/default.yaml.

Each solve archives argv, resolved YAML config, terminal capture, and solution
copies under ``<output_dir>/runs/<run_id>/`` unless disabled (see ``--no-run-archive``).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Make 'src' importable when running as python -m solver from repo root ──
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from solver.data import load_problem, print_validation_report
from solver.ga import solve_all_weeks, FitnessEvaluator
from solver.output import (
    build_week_solution,
    write_solution_json,
    write_summary_csv,
    print_solution_summary,
    solution_fully_feasible,
)
from solver.run_archive import RunArchive, collect_explicit_cli, stable_config_hash


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def config_hash(cfg: dict) -> str:
    """Stable hash of the effective config (delegates to run_archive)."""
    return stable_config_hash(cfg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m solver",
        description="Interfor Dedicated Truck Routing — GA Solver",
    )
    p.add_argument("--data", type=Path,
                   help="Path to 'AIMMS-MOPTA Interfor data.xlsx'")
    p.add_argument("--config", type=Path,
                   default=Path("configs/default.yaml"),
                   help="YAML config file (default: configs/default.yaml)")
    p.add_argument("--seed", type=int, help="RNG seed (overrides config)")
    p.add_argument("--generations", type=int,
                   help="Max GA generations (overrides config)")
    p.add_argument("--pop", type=int,
                   help="Population size (overrides config)")
    p.add_argument("--mutation", type=float,
                   help="Per-gene mutation probability 0..1 (overrides ga.mutation_rate_gene)")
    p.add_argument("--elite-fraction", type=float, dest="elite_fraction",
                   help="Fraction in (0,1] of population that is elite; overrides ga.elite_size")
    p.add_argument("--eval-workers", type=int,
                   help="Fitness evaluation workers; 1 disables parallelism, 0 = auto")
    p.add_argument("--edu-workers", type=int,
                   help="Education workers; 1 disables parallelism, 0 = auto")
    p.add_argument("--subset", type=int,
                   help="Limit to first N orders per week (debug mode)")
    p.add_argument("--output-dir", type=Path,
                   help="Directory for output files")
    p.add_argument("--validate-only", action="store_true",
                   help="Only load and validate data, then exit")
    p.add_argument("--no-education", action="store_true",
                   help="Disable LS education (use GA-only, no intra-route local search)")
    p.add_argument("--no-run-archive", action="store_true",
                   help="Do not write per-run folder under output/runs/ (no tee / meta copy)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log = logging.getLogger("solver.__main__")

    cfg = load_config(args.config)
    cfg.setdefault("ga", {})
    cfg.setdefault("constraints", {})
    cfg.setdefault("penalties", {})
    cfg.setdefault("data", {})
    cfg.setdefault("output", {})
    cfg.setdefault("education", {})

    # CLI overrides
    if args.seed is not None:
        cfg["ga"]["seed"] = args.seed
    if args.generations is not None:
        cfg["ga"]["max_generations"] = args.generations
    if args.pop is not None:
        cfg["ga"]["population_size"] = args.pop
    if args.mutation is not None:
        if not 0.0 <= args.mutation <= 1.0:
            # Logging not configured yet
            print("--mutation must be between 0 and 1 (got %s)" % (args.mutation,), file=sys.stderr)
            sys.exit(2)
        cfg["ga"]["mutation_rate_gene"] = args.mutation
    if args.elite_fraction is not None:
        if not 0.0 < args.elite_fraction <= 1.0:
            print(
                "--elite-fraction must be in (0, 1] (got %s)" % (args.elite_fraction,),
                file=sys.stderr,
            )
            sys.exit(2)
        cfg["ga"]["elite_fraction"] = args.elite_fraction
    if args.eval_workers is not None:
        if args.eval_workers < 0:
            print("--eval-workers must be >= 0 (got %s)" % (args.eval_workers,), file=sys.stderr)
            sys.exit(2)
        cfg["ga"]["eval_workers"] = args.eval_workers
    if args.edu_workers is not None:
        if args.edu_workers < 0:
            print("--edu-workers must be >= 0 (got %s)" % (args.edu_workers,), file=sys.stderr)
            sys.exit(2)
        cfg["education"]["workers"] = args.edu_workers
    if args.subset is not None:
        cfg["ga"]["order_subset"] = args.subset
    if args.no_education:
        cfg["education"]["enabled"] = False

    data_path = args.data or Path(cfg["data"].get(
        "path", "AIMMS_MOPTA_2026/AIMMS-MOPTA Interfor data.xlsx"
    ))
    output_dir = args.output_dir or Path(cfg["output"].get("directory", "output"))

    run_archive_on = (
        cfg["output"].get("run_archive", True)
        and not args.no_run_archive
    )
    runs_subdir = str(cfg["output"].get("runs_subdirectory", "runs"))
    config_path_resolved = args.config.resolve()
    explicit_cli = collect_explicit_cli(args)

    archive: RunArchive | None = None
    if run_archive_on:
        archive = RunArchive(
            output_dir=output_dir,
            cfg=cfg,
            argv=list(sys.argv),
            config_path=config_path_resolved,
            data_path=data_path.resolve(),
            runs_subdir=runs_subdir,
        )
        archive.start()

    setup_logging(args.log_level)

    exit_code = 0
    solution: dict | None = None
    feasible: bool | None = None
    primary_json: Path | None = None
    primary_csv: Path | None = None
    debug_json: Path | None = None

    try:
        try:
            ch = config_hash(cfg)
            log.info("Config hash: %s", ch)
            log.info("Data path: %s", data_path)
            if archive is not None:
                log.info("Run archive: %s", archive.run_dir)

            # ── Load data ─────────────────────────────────────────────────
            t_load = time.time()
            problem = load_problem(data_path, cfg_sheets=cfg.get("data", {}).get("sheets"))
            log.info("Data loaded in %.2fs", time.time() - t_load)

            print_validation_report(problem)

            if args.validate_only:
                log.info("--validate-only flag set. Exiting after validation.")
            else:
                # ── Run GA per week ───────────────────────────────────────────
                t_solve = time.time()
                week_results = solve_all_weeks(problem, cfg, verbose=True)
                elapsed = time.time() - t_solve
                log.info("Total solve time: %.1f seconds", elapsed)

                # ── Build output ──────────────────────────────────────────────
                solution = {"weeks": [], "meta": {}}
                total_cost = 0.0

                for week_label, (best_chrom, best_fitness, history, baseline_T, baseline_G) in sorted(
                    week_results.items()
                ):
                    week_orders = problem.orders_by_week[week_label]
                    subset = cfg["ga"].get("order_subset", None)
                    week_order_indices = np.array([o.order_idx for o in week_orders])
                    if subset is not None and subset < len(week_order_indices):
                        week_order_indices = week_order_indices[:subset]

                    week_sol = build_week_solution(
                        week_label=week_label,
                        best_chrom=best_chrom,
                        week_order_indices=week_order_indices,
                        problem=problem,
                        cfg_constraints=cfg["constraints"],
                    )
                    week_sol["ga_history"] = history
                    week_sol["baselines"] = {
                        "T_all_tender": round(baseline_T, 2),
                        "G_greedy": round(baseline_G, 2),
                        "C_ga_best": round(best_fitness, 2),
                    }
                    solution["weeks"].append(week_sol)
                    total_cost += week_sol["totals"]["grand_total"]

                solution["meta"] = {
                    "solve_time_seconds": round(elapsed, 2),
                    "total_cost_all_weeks": round(total_cost, 2),
                    "config_hash": ch,
                    "n_weeks": len(week_results),
                    "seed": cfg["ga"].get("seed", 42),
                }

                print_solution_summary(solution)

                sol_json_name = cfg["output"].get("solution_json", "solution.json")
                sol_csv_name = cfg["output"].get("summary_csv", "solution_summary.csv")
                debug_name = cfg["output"].get("infeasible_debug_json", "last_infeasible_debug.json")

                if not solution_fully_feasible(solution):
                    log.error(
                        "Hard feasibility check failed; not writing %s / %s.",
                        sol_json_name,
                        sol_csv_name,
                    )
                    debug_json = write_solution_json(solution, output_dir, debug_name)
                    log.error("Diagnostics written to %s", debug_json)
                    feasible = False
                    primary_json = debug_json
                    exit_code = 1
                else:
                    feasible = True
                    primary_json = write_solution_json(solution, output_dir, sol_json_name)
                    primary_csv = write_summary_csv(solution, output_dir, sol_csv_name)
                    log.info("Done. Total cost: $%.2f  Solve time: %.1fs", total_cost, elapsed)

        except Exception:
            exit_code = 1
            raise

    finally:
        if archive is not None:
            archive.finish(
                config_hash=config_hash(cfg),
                explicit_cli=explicit_cli,
                validate_only=args.validate_only,
                solution=solution,
                feasible=feasible,
                primary_sol_json=primary_json,
                primary_sol_csv=primary_csv,
                debug_json_path=debug_json,
                exit_code=exit_code,
            )

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

# Interfor Dedicated Truck Routing — GA Solver

Prototype solver for the AIMMS–MOPTA 2026 Interfor competition.  
Decides which of 1,000 orders go to 20 dedicated trucks vs. the open market,
and sequences each truck's multi-stop route to minimise total freight cost.

---

## Quick start

```bash
# 1. From repository root: create venv with Python 3.13 (same as fhathorn/mopta-26 starter:
#    AIMMS_MOPTA_2026/pyproject.toml sets requires-python = "==3.13.*")
py -3.13 -m venv .venv
# Windows: .venv\Scripts\activate.bat (cmd)  or  .venv\Scripts\Activate.ps1 (PowerShell)
# Unix:    source .venv/bin/activate

# 2. Install solver + test dependencies
pip install -r requirements.txt

# 3. Run solver (full 1,000 orders, default config)
python src/solver/__main__.py \
    --data "AIMMS_MOPTA_2026/AIMMS-MOPTA Interfor data.xlsx"

# 4. Debug run (50 orders/week, 50 generations)
python src/solver/__main__.py \
    --data "AIMMS_MOPTA_2026/AIMMS-MOPTA Interfor data.xlsx" \
    --subset 50 --pop 30 --generations 50 --seed 0

# 4b. Same debug run, but parallelize the within-generation hotspots
python src/solver/__main__.py \
    --data "AIMMS_MOPTA_2026/AIMMS-MOPTA Interfor data.xlsx" \
    --subset 100 --pop 60 --generations 80 --seed 42 \
    --eval-workers 8 --edu-workers 8

# 5. Data validation only
python src/solver/__main__.py \
    --data "AIMMS_MOPTA_2026/AIMMS-MOPTA Interfor data.xlsx" \
    --validate-only

# 6. Run tests
python -m pytest tests/ -v
```

All outputs land in `output/solution.json` and `output/solution_summary.csv`.

**Run archive (audit trail):** each invocation (including `--validate-only`) can write a folder  
`<output_dir>/<runs_subdirectory>/<run_id>/` with:

- `terminal.log` — everything printed to stdout/stderr after the archive starts  
- `meta.json` — argv, cwd, config path, resolved `config_hash`, explicit CLI overrides, `solution_meta`, exit code, optional `git_rev`  
- `resolved_config.yaml` — **effective** config after YAML + CLI overrides  
- `solution.json` / `solution_summary.csv` — copies when the solve finished **feasible**  
- `last_infeasible_debug.json` — copy when feasibility failed  

A one-line index is appended to `<output_dir>/<runs_subdirectory>/index.jsonl`.  
Disable with **`--no-run-archive`** or `output.run_archive: false` in YAML.

Older hand-saved transcripts (pre–run-archive) live under [`docs/saved-runs/`](saved-runs/) (for example [`docs/saved-runs/2026-04-18-pop1000-noedu-500gens/`](saved-runs/2026-04-18-pop1000-noedu-500gens/)).

If you still have an old **`AIMMS_MOPTA_2026/.venv`**, you can remove it after moving installs to **repo-root `.venv`** so only one environment is active.

---

## Method choice

**Memetic GA on a flat integer assignment chromosome**

| Approach | Pros | Cons | Chosen? |
|---|---|---|---|
| Plain GA (permutation) | Simple | O(n²) crossover, naive VRP | No |
| DEAP/pymoo | Powerful, many operators | Heavy dependency | No |
| Our choice: hand-rolled GA + NN heuristic | Minimal deps, fast, tunable | Route order not evolved | **Yes** |
| Large Neighbourhood Search | Best for pure VRP | Overkill for assignment layer | Future |

The key insight: the objective's **assignment layer** (which orders go to which truck)
dominates cost variance. The **route-within-truck** layer is handled by a
nearest-neighbour heuristic (minimises deadhead), which is a fast local-optima
approximation sufficient for the repair operator.

---

## Chromosome encoding

**Layer A — Assignment vector** (evolved):
```
chrom[i] ∈ {-1, 0 … 19}   for i = 0 … n_orders-1
  -1  = order tendered to open market
   k  = order assigned to dedicated truck k
```
One gene per order in the planning week. Length 450–550 per week.

**Layer B — Route order** (derived, not evolved):
For each truck, the sequence of its assigned orders is determined by a
**greedy nearest-neighbour** heuristic: starting from truck home, at each step
pick the unserved order whose *origin* is nearest (by lane mileage) to the
current location.

This keeps the chromosome compact (int8 × ~500 = ~500 bytes per individual)
and fitness evaluation fast (~35 ms/individual on full 550-order week).

---

## Cost model

```
total_cost = dedicated_cost + tender_cost

dedicated_cost = Σ_k  max(earnings_k, $4,000)
  where earnings_k = Σ_{i assigned to k} Lane.Cost(origin_i, dest_i)

tender_cost = Σ_{i tendered}  Lane.Cost(origin_i, dest_i)
                             + DifficultLane.adder_per_mile × Lane.Mileage(origin_i, dest_i)
                               [if a matching Difficult Lane row exists]
```

**Difficult lane matching** (`geography.py::matches_destination_key`):
- Two-letter code `TX` → match any order destination in state TX (from `CITY,ST` suffix)
- `GA_303` → match destination cities whose 3-digit ZIP prefix is `303`
  (hardcoded dict `CITY_ZIP3` in `geography.py` covers all 218 dataset cities)

---

## Constraints modelled

| Constraint | Hard check | Soft penalty (GA) |
|---|---|---|
| Max 450 miles/day | ✓ (scheduler) | — (scheduler enforces by construction) |
| Max 50% deadhead | ✓ | ✓ |
| Final delivery ≤ 100 mi from home | ✓ | ✓ |
| ≥ 2 consecutive days off (≤ 5 working days) | ✓ | ✓ |
| Earnings ≥ $4,000/week | ✓ | Modelled as cost min (max rule) |

The **repair operator** (`feasibility.py::repair_truck_route`) ejects orders
until each active truck passes **all** hard checks (weekly miles, deadhead,
**$4,000 earnings**, **≤100 mi final return to home**, daily miles, working
days). Ejected orders go to tender. It runs periodically during the GA **and**
once on the **best** chromosome after the GA finishes (elites do not receive
the periodic pass). The CLI writes `solution.json` / `solution_summary.csv`
**only** if every week is feasible; otherwise it writes
`last_infeasible_debug.json` and exits with code **1**.

---

## Performance (measured on dev machine)

| Instance | Orders | Pop | Gens | Runtime |
|---|---|---|---|---|
| Subset debug | 50/week | 30 | 50 | 0.3 s |
| Full instance | 450+550 | 60 | ~120–270 | **11.7 s** |

Scales linearly: 300 generations on full instance ≈ 25–30 s.  
Well under the 8-hour competition target.

---

## Repository layout

```
src/solver/
  __init__.py
  __main__.py       # CLI entry point
  data.py           # Excel loader, dataclasses, ProblemData
  geography.py      # ZIP-prefix matching, difficult-lane lookup, Haversine
  cost.py           # Lane cost, surcharge, total cost for a chromosome
  scheduler.py      # Nearest-neighbour route, day-assignment (450 mi/day)
  feasibility.py    # Constraint checks + soft penalties + repair operator
  ga.py             # GA main loop, operators, multi-week driver
  output.py         # solution.json + solution_summary.csv writer

tests/
  test_geography.py   # 27 tests: ZIP matching, state matching, Haversine
  test_cost.py        # 12 tests: earnings, minimum guarantee, surcharge rules
  test_feasibility.py # 15 tests: each constraint, toy end-to-end schedule
  test_data.py        # 18 integration tests against real Excel (skipped if no file)

configs/
  default.yaml        # All tunable parameters

assumptions.yaml      # All modelling assumptions and known gaps
```

---

## Known gaps vs. competition PDF

| Gap | Status |
|---|---|
| Driver pay model not specified in PDF | Assumed = sum of Lane.Cost (confirmed direction by John Cox) |
| ZIP codes for 218 cities not in data | Hardcoded dict in `geography.py::CITY_ZIP3`; `AUGUSTA,GA` added after data inspection found it in difficult-lane destinations |
| Truck capacity (weight/volume) | Not in data; treated as uncapped |
| Service time (loading/unloading) | Not in data; assumed zero |
| Route-order evolution | Not evolved; nearest-neighbour heuristic used |
| $4,000 minimum modelled as cost, not pure constraint | Conservative: Interfor pays at least $4,000/active truck |

See `assumptions.yaml` for full detail.

---

## Reporting, baselines & tuning

For competition-style evidence (not “we ran once and it felt good”), follow the repo prompts:

- **Baselines T / G / C**, **multi-seed (15–30)**, and **how to tune `configs/default.yaml`** (feasible export first, then cost vs baselines, then trim runtime): [genetic-algorithm-implementation-prompt.md](genetic-algorithm-implementation-prompt.md) — § *Solution quality* and § *Tuning `configs/default.yaml`*.
- **Phase 2 (memetic):** same baselines + multi-seed when comparing education on/off — [genetic-algorithm-phase2-memetic-prompt.md](genetic-algorithm-phase2-memetic-prompt.md) — § *Baselines + multi-seed*.

Log **T**, **G**, **C** and seed statistics in this file or under `results/` as you iterate.

---

## CLI reference

```
python src/solver/__main__.py [OPTIONS]

  --data PATH          Path to Excel workbook
  --config PATH        YAML config (default: configs/default.yaml)
  --seed INT           RNG seed
  --generations INT    Max GA generations
  --pop INT            Population size
  --mutation FLOAT     Per-gene mutation probability in [0, 1] (ga.mutation_rate_gene)
  --eval-workers INT   Fitness workers; 1 = sequential, 0 = auto
  --edu-workers INT    Education workers; 1 = sequential, 0 = auto
  --subset INT         Limit to first N orders per week (debug)
  --output-dir PATH    Output directory (default: output/)
  --validate-only      Data check only, no GA
  --no-education       Disable LS education (GA-only; use for A/B comparison)
  --no-run-archive     Skip per-run folder under output/runs/ (no tee / meta)
  --log-level LEVEL    DEBUG|INFO|WARNING|ERROR
```

Parallelism stays inside each generation. The overall GA loop is still serial. `workers=1` preserves the sequential path; `0` means "auto", capped at 12 workers on the dev machine target.

---

## Phase 2 — Memetic GA (LS education)

### Intra-route local search operators

Three first-improvement operators in `src/solver/education.py`, applied per truck after crossover/mutation:

| Operator | Move |
|---|---|
| Or-opt-1 | Relocate one order to the best other position |
| 2-opt | Reverse sub-sequence [i..j] |
| Or-opt-2 | Relocate a consecutive pair to the best position |

**LS objective**: `total_deadhead_miles + home_penalty * max(0, final_return_dist - 100)`

### Route-key encoding

Each individual carries a `float32[n_orders]` keys array. For truck *t*, assigned orders are sorted by key value (lower = earlier in route).

- Keys are initialised from NN routes so generation-0 routes are already good.
- After **mutation** and after **repair**, keys are **re-built from NN** for the current assignment (mutated genes would otherwise desync keys from the chromosome).
- After education, keys encode the LS-improved permutation.
- Crossover can mix parental keys, but offspring keys are always re-aligned to the child assignment before evaluation.

### Reproducibility and workers

- `ga.seed` now drives route-key initialisation as well as the GA operators, so `--eval-workers 1 --edu-workers 1` is deterministic.
- Parallel fitness evaluation is still deterministic because each individual is evaluated independently and results are merged in population order.
- Parallel education uses per-individual seeds derived from `(seed, generation, index)` rather than shared RNG state, so enabling workers changes runtime behaviour but does not introduce hidden cross-task randomness.

### Strict feasibility export contract

`solution.json` / `solution_summary.csv` are written **only** when every week passes all five hard constraints. On failure: `last_infeasible_debug.json` is written and the process exits with code 1.

### Baselines T / G / C

| Baseline | Definition |
|---|---|
| **T** | All orders tendered; no dedicated trucks |
| **G** | Difficult-lane orders assigned to nearest-home truck; rest tendered |
| **C** | GA best after repair + final education |

Logged per week and stored in `solution.json` under `baselines`.

### A/B comparison: education on vs. off

```bash
python src/solver/__main__.py --data "..." --seed 42              # education on (default)
python src/solver/__main__.py --data "..." --seed 42 --no-education  # education off
```

**Multi-seed (recommended):** full 1000 orders, `configs/default.yaml`, seeds **42–53** (12 seeds), same machine (Apr 2026). Per-run `solution.json` copies also under `output/ab_sweep/` (gitignored). Versioned aggregate TSV: [`docs/ab-sweep-education-seeds42-53.tsv`](ab-sweep-education-seeds42-53.tsv).

| Aggregate | Education **on** | `--no-education` |
|---|---|---|
| Mean grand total (12 seeds) | **$1,358,091** | $1,360,722 |
| Population stdev of grand total | $1,946 | $1,876 |
| Mean wall time (solve only, 2 weeks) | **~36 s** | ~17 s |
| Seeds with **lower** grand total | **10** | 2 |

Mean difference (on − off) ≈ **−$2,631** (education **cheaper on average**). Seed **42** is an outlier in the other direction (off wins by ~$181); single-seed A/B can mis-rank the modes.

| Seed 42 only (illustrative) | Week 1 | Week 2 | Grand total | Solve |
|---|---|---|---|---|
| education=on | 598,536 | 758,145 | 1,356,681 | ~34 s |
| education=off | 596,618 | 759,881 | 1,356,500 | ~30 s |

### Tuning `education:` in `configs/default.yaml`

| Key | Role |
|---|---|
| `every_n_generations` | Run an LS pass only every *N* generations (reduces wall time). |
| `p_edu` | Chance each **non-elite** is educated when a pass runs (lower → faster, less LS). |
| `max_ls_iters` | Operator sweep depth per truck per educate call. |
| `home_penalty_per_mile` | LS surrogate for final return > 100 mi; too high can fight the freight objective. |

---

## AIMMS integration path

This solver is designed for later integration via the **AIMMS Python Bridge**:

1. `data.py::load_problem()` can be called from a bridge script instead of Excel.
2. `ga.py::solve_all_weeks()` returns structured dicts — trivially JSON-serialisable.
3. `output.py::build_week_solution()` produces the per-carrier routing plan that
   the AIMMS UI reads (carrier name, ordered loads, days, costs).
4. Parameters (`min_weekly_earnings`, `fleet_size`) are in `configs/default.yaml`
   and can be overridden by the AIMMS user-parameter layer.

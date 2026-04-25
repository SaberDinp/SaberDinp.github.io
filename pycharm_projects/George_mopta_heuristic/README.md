# MOPTA 2026 — Interfor dedicated truck routing

Python GA / memetic solver, official workbook and AIMMS starter (`AIMMS_MOPTA_2026/`), `pytest` suite, and a **reference feasible solution** (`output/solution.json`, `output/solution_summary.csv`). See `docs/python-ga-solver.md` for design and CLI details; `docs/AIMMS-MOPTA_2026_Competition.pdf` is the problem statement.

## Contents

| Path | Purpose |
|------|---------|
| `src/solver/` | GA + local-search solver |
| `configs/` | YAML defaults |
| `requirements.txt` | pip dependencies |
| `tests/` | pytest |
| `AIMMS_MOPTA_2026/` | Competition Excel + AIMMS project |
| `output/solution.json` | Full solution (structured) |
| `output/solution_summary.csv` | Per-order table (1000 rows + header) |
| `output/runs/<run_id>/` | Example archived run: log, `meta.json`, `resolved_config.yaml`, solution copies |
| `docs/python-ga-solver.md` | Solver documentation |
| `docs/AIMMS-MOPTA_2026_Competition.pdf` | Problem PDF |

---

## 1. Prerequisites

- **Python 3.13** (matches `AIMMS_MOPTA_2026/pyproject.toml`).
- Network for initial `pip install`.

```powershell
python --version
```

Expect `Python 3.13.x`. Install from [python.org](https://www.python.org/downloads/) if needed.

---

## 2. Virtual environment

Use the directory that contains `src/`, `requirements.txt`, and `AIMMS_MOPTA_2026/`.

**Windows (PowerShell):**

```powershell
cd <path-to-this-folder>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation fails: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use **cmd** with `.venv\Scripts\activate.bat`.

**macOS / Linux:**

```bash
cd /path/to/this-folder
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Keep the venv activated for the commands below.

**Optional:**

```powershell
python -c "import numpy, pandas, yaml; print('OK')"
pytest tests -q
```

---

## 3. Reference solution (no solve)

Open **`output/solution.json`** and **`output/solution_summary.csv`** for the current reference plan. **`solution.json`** → `"meta"` has `total_cost_all_weeks`, `seed`, `config_hash`.

Example archive: **`output/runs/20260418T143623Z_c9a7e463/`** (~**$1,349,730.66** total, feasible both weeks). See `meta.json` and `terminal.log` for argv and timings.

---

## 4. Run the solver (match that reference)

From the same directory (venv on):

```powershell
python src/solver/__main__.py --data "AIMMS_MOPTA_2026/AIMMS-MOPTA Interfor data.xlsx" --pop 1000 --elite-fraction 0.1 --eval-workers 0 --edu-workers 0 --seed 42
```

- **`--seed 42`**: default in `configs/default.yaml`; explicit here for clarity.
- **`--eval-workers 0`** / **`--edu-workers 0`**: auto parallel pools. Use **`1`** / **`1`** if you want sequential evaluation for easier log-to-log comparison.

Writes **`output/solution.json`**, **`output/solution_summary.csv`**, and usually **`output/runs/<run_id>/`** (disable with `--no-run-archive`). Re-running replaces those outputs; copy them first if you want to keep the current files.

---

## 5. Links

- https://coral.ise.lehigh.edu/mopta2026/competition/
- https://github.com/fhathorn/mopta-26

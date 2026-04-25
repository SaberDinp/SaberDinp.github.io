"""
run_archive.py — Per-invocation run log: argv, resolved config, terminal capture, artifacts.

Archives live under ``<output_dir>/<runs_subdirectory>/<run_id>/``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO

import yaml


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stable_config_hash(cfg: dict) -> str:
    """Same definition as ``solver.__main__.config_hash`` (must stay in sync)."""
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _git_head() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path.cwd(),
        )
        if r.returncode == 0:
            return r.stdout.strip()[:40]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


class TeeTextIO(TextIO):
    """Write to a primary stream and a log file (UTF-8)."""

    def __init__(self, primary: TextIO, log_fp: TextIO) -> None:
        self._primary = primary
        self._log = log_fp

    def write(self, s: str) -> int:
        n = self._primary.write(s)
        self._primary.flush()
        self._log.write(s)
        self._log.flush()
        return n

    def flush(self) -> None:
        self._primary.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    def fileno(self) -> int:
        return self._primary.fileno()

    def writable(self) -> bool:
        return True


@dataclass
class RunArchive:
    """
    Create ``run_dir``, tee stdout/stderr to ``terminal.log``, then on ``finish()``
    write ``meta.json``, ``resolved_config.yaml``, copy solution artifacts, append ``index.jsonl``.
    """

    output_dir: Path
    cfg: dict
    argv: list[str]
    config_path: Path
    data_path: Path
    runs_subdir: str = "runs"
    _log_fp: Optional[TextIO] = field(default=None, repr=False)
    _saved_stdout: Optional[TextIO] = field(default=None, repr=False)
    _saved_stderr: Optional[TextIO] = field(default=None, repr=False)
    run_id: str = field(init=False)
    run_dir: Path = field(init=False)
    started_at: str = field(init=False)
    _config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self._config_hash = stable_config_hash(self.cfg)
        self.run_id = f"{_utc_stamp()}_{self._config_hash[:8]}"
        self.run_dir = (self.output_dir / self.runs_subdir / self.run_id).resolve()
        self.started_at = datetime.now(timezone.utc).isoformat()

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.run_dir / "terminal.log"
        self._log_fp = open(log_path, "w", encoding="utf-8", newline="")
        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr
        sys.stdout = TeeTextIO(self._saved_stdout, self._log_fp)
        sys.stderr = TeeTextIO(self._saved_stderr, self._log_fp)

    def stop_tee_only(self) -> None:
        if self._saved_stdout is not None:
            sys.stdout = self._saved_stdout
        if self._saved_stderr is not None:
            sys.stderr = self._saved_stderr
        if self._log_fp is not None:
            self._log_fp.close()
            self._log_fp = None
        self._saved_stdout = None
        self._saved_stderr = None

    def finish(
        self,
        *,
        config_hash: str,
        explicit_cli: Mapping[str, Any],
        validate_only: bool,
        solution: Optional[dict],
        feasible: Optional[bool],
        primary_sol_json: Optional[Path],
        primary_sol_csv: Optional[Path],
        debug_json_path: Optional[Path],
        exit_code: int,
    ) -> Path:
        """
        Restore streams, write meta + resolved YAML, copy artifacts, append index line.
        Call once at process exit (success or failure).
        """
        self.stop_tee_only()

        finished_at = datetime.now(timezone.utc).isoformat()
        meta: dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "exit_code": exit_code,
            "cwd": str(Path.cwd().resolve()),
            "argv": list(self.argv),
            "config_path": str(self.config_path.resolve()),
            "data_path": str(self.data_path.resolve()),
            "output_dir": str(self.output_dir.resolve()),
            "run_dir": str(self.run_dir),
            "config_hash": config_hash,
            "git_rev": _git_head(),
            "validate_only": validate_only,
            "explicit_cli_overrides": dict(explicit_cli),
            "feasible": feasible,
        }
        if solution is not None:
            meta["solution_meta"] = solution.get("meta", {})

        with open(self.run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        with open(self.run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False, default_flow_style=False)

        if not validate_only and solution is not None:
            if feasible and primary_sol_json is not None and primary_sol_json.is_file():
                shutil.copy2(primary_sol_json, self.run_dir / "solution.json")
                if primary_sol_csv is not None and primary_sol_csv.is_file():
                    shutil.copy2(primary_sol_csv, self.run_dir / "solution_summary.csv")
            elif debug_json_path is not None and debug_json_path.is_file():
                shutil.copy2(debug_json_path, self.run_dir / "last_infeasible_debug.json")

        index_line = {
            "run_id": self.run_id,
            "finished_at": finished_at,
            "config_hash": config_hash,
            "feasible": feasible,
            "validate_only": validate_only,
            "total_cost": (solution or {}).get("meta", {}).get("total_cost_all_weeks"),
            "argv_one_line": " ".join(self.argv),
        }
        index_path = self.output_dir / self.runs_subdir / "index.jsonl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(index_line, default=str) + "\n")

        return self.run_dir


def collect_explicit_cli(args: Any) -> dict[str, Any]:
    """Which CLI knobs were explicitly set (for meta.json)."""
    out: dict[str, Any] = {}
    if args.seed is not None:
        out["seed"] = args.seed
    if args.generations is not None:
        out["generations"] = args.generations
    if args.pop is not None:
        out["pop"] = args.pop
    if args.mutation is not None:
        out["mutation"] = args.mutation
    if getattr(args, "elite_fraction", None) is not None:
        out["elite_fraction"] = args.elite_fraction
    if args.subset is not None:
        out["subset"] = args.subset
    if getattr(args, "eval_workers", None) is not None:
        out["eval_workers"] = args.eval_workers
    if getattr(args, "edu_workers", None) is not None:
        out["edu_workers"] = args.edu_workers
    if args.data is not None:
        out["data"] = str(args.data)
    if getattr(args, "config", None) is not None:
        out["config"] = str(args.config)
    if args.output_dir is not None:
        out["output_dir"] = str(args.output_dir)
    if args.no_education:
        out["no_education"] = True
    if getattr(args, "no_run_archive", False):
        out["no_run_archive"] = True
    if args.validate_only:
        out["validate_only"] = True
    return out

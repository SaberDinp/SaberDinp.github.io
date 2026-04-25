"""Tests for solver.run_archive (tee + meta layout)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

from solver.run_archive import RunArchive, TeeTextIO, collect_explicit_cli, stable_config_hash


def test_stable_config_hash_deterministic():
    cfg = {"ga": {"seed": 1}, "b": [1, 2]}
    assert stable_config_hash(cfg) == stable_config_hash(dict(cfg))


def test_tee_writes_to_both_streams():
    primary = io.StringIO()
    log_fp = io.StringIO()
    tee = TeeTextIO(primary, log_fp)
    n = tee.write("hello")
    assert n == 5
    tee.flush()
    assert primary.getvalue() == "hello"
    assert log_fp.getvalue() == "hello"


def test_run_archive_finish_writes_meta_and_yaml(tmp_path: Path):
    out = tmp_path / "out"
    cfg = {"ga": {"seed": 99}, "output": {"directory": str(out)}}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    data_path = tmp_path / "fake.xlsx"
    data_path.write_bytes(b"")

    ar = RunArchive(
        output_dir=out,
        cfg=cfg,
        argv=["py", "solver"],
        config_path=cfg_path,
        data_path=data_path,
        runs_subdir="runs",
    )
    ar.run_dir.mkdir(parents=True, exist_ok=True)
    ar.stop_tee_only()  # skip tee; we only test finish artifact writes

    sol = {"meta": {"total_cost_all_weeks": 123.45, "seed": 99}}
    primary_json = tmp_path / "sol.json"
    primary_json.write_text(json.dumps(sol), encoding="utf-8")

    ar.finish(
        config_hash=stable_config_hash(cfg),
        explicit_cli={"seed": 99},
        validate_only=False,
        solution=sol,
        feasible=True,
        primary_sol_json=primary_json,
        primary_sol_csv=None,
        debug_json_path=None,
        exit_code=0,
    )

    meta = json.loads((ar.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["exit_code"] == 0
    assert meta["explicit_cli_overrides"]["seed"] == 99
    assert meta["solution_meta"]["total_cost_all_weeks"] == 123.45
    assert (ar.run_dir / "resolved_config.yaml").is_file()
    assert (ar.run_dir / "solution.json").is_file()
    idx_lines = (out / "runs" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(idx_lines) == 1
    assert "123.45" in idx_lines[0]


def test_collect_explicit_cli_no_run_archive():
    class A:
        seed = None
        generations = None
        pop = None
        mutation = None
        subset = None
        data = None
        config = Path("x.yaml")
        output_dir = None
        no_education = False
        no_run_archive = True
        validate_only = False

    d = collect_explicit_cli(A())
    assert d.get("no_run_archive") is True

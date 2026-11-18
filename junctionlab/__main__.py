"""Command line: python -m junctionlab {list,run} ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from junctionlab.examples import EXAMPLES_DIR, list_examples, load_example
from junctionlab.runner import run_scenario
from junctionlab.scenario import Scenario


def _load(target: str) -> Scenario:
    """An example key, an example/draft JSON file, or a full scenario JSON file."""
    path = Path(target)
    if not path.suffix:
        path = EXAMPLES_DIR / f"{target}.json"
    if not path.is_file():
        keys = ", ".join(e.key for e in list_examples())
        sys.exit(f"Not found: {target}. Examples: {keys}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "draft" in data:
        return load_example(path).scenario()
    return Scenario.model_validate(data)


def _describe(s: Scenario) -> None:
    print(f"Scenario: {s.name} ({s.junction_type}, {s.control.kind})")
    for arm, a in s.approaches.items():
        lanes = " | ".join("+".join(m[0].upper() for m in lane) for lane in a.lanes)
        bus = f" + bus lane ({a.bus_demand_bph:g} buses/h)" if a.bus_lane else ""
        print(f"  {arm:<5} {a.demand_vph:>6g} veh/h  lanes [{lanes}]{bus}")
    if s.control.kind == "signal":
        greens = " / ".join(f"{g:g}" for g in s.control.green_s)
        print(f"  signal: cycle {s.control.cycle_s:g} s, {s.control.phase_plan}, greens {greens} s")
    if s.assumed:
        print(f"  assumed (defaults): {len(s.assumed)} field(s)")


def cmd_run(args: argparse.Namespace) -> int:
    scenario = _load(args.target)
    _describe(scenario)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    result = run_scenario(scenario, run_dir=args.out, seeds=seeds)

    print(f"\nRun directory: {result.dir}")
    print(f"{'seed':>4}  {'ended by':<24}{'sim end':>8}{'inserted':>10}{'arrived':>9}  warnings")
    any_warning = False
    for r in result.seeds:
        warnings = "; ".join(r.warnings) or "-"
        any_warning |= bool(r.warnings)
        print(
            f"{r.seed:>4}  {r.ended_by:<24}{r.end_time_s:>7.0f}s{r.vehicles_inserted:>10}"
            f"{r.vehicles_arrived:>9}  {warnings}"
        )
    print("\nPer-seed outputs: tripinfo.xml, e2.xml, statistics.xml, sumo.log, run.json")
    print("Delay, queue and LOS reporting comes in M3.")
    return 1 if any_warning and args.strict else 0


def cmd_list(_: argparse.Namespace) -> int:
    for e in list_examples():
        print(f"{e.key:<22} {e.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="junctionlab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list bundled examples").set_defaults(func=cmd_list)
    run = sub.add_parser("run", help="simulate an example or scenario file")
    run.add_argument("target", help="example key (see 'list') or path to a JSON file")
    run.add_argument("--seeds", help="comma-separated seeds (default: config.yaml)")
    run.add_argument("--out", type=Path, help="run directory (default: runs/<time>_<name>)")
    run.add_argument("--strict", action="store_true", help="exit 1 if any run has a warning")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Command line: python -m junctionlab {list,run,webster,interpret,agent,agent-eval} ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from junctionlab.baseline import by_movement as baseline_by_movement
from junctionlab.examples import EXAMPLES_DIR, list_examples, load_example
from junctionlab.metrics import Stat, Summary
from junctionlab.runner import run_scenario
from junctionlab.scenario import Scenario, UnconfirmedDemand


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


def _print_unconfirmed(s: Scenario) -> None:
    print("\nNot run: the demand below is the model's reading of the text.")
    for path in s.needs_confirmation:
        print(f"  {path} = {_value(s, path):g}: {s.interpreted[path]}")
    print("Correct the description, or repeat with --confirm-demand to accept this reading.")


def _value(s: Scenario, path: str) -> float:
    _, arm, name = path.split(".")
    return getattr(s.approaches[arm], name)


def _fmt(stat: Stat | None, digits: int = 1) -> str:
    if stat is None:
        return "-"
    return f"{stat.mean:.{digits}f} ({stat.min:.{digits}f}-{stat.max:.{digits}f})"


def print_summary(summary: Summary) -> None:
    print(f"\nResults over {len(summary.seeds)} seed(s): mean (min-max)")
    header = (
        f"{'approach':<9}{'control delay s/veh':>25}{'total delay s/veh':>22}"
        f"{'max queue m':>16}{'max queue veh':>15}{'throughput veh/h':>20}   LOS (seeds)"
    )
    rule = "-" * (len(header) + 4)
    print(header)
    print(rule)
    for a in summary.approaches.values():
        star = "*" if a.delay_lower_bound else " "
        print(
            f"{a.arm:<9}{_fmt(a.control_delay_s) + star:>25}{_fmt(a.delay_s) + star:>22}"
            f"{_fmt(a.queue_m, 0):>16}{_fmt(a.queue_veh, 0):>15}"
            f"{_fmt(a.throughput_vph, 0):>20}   {a.los} ({a.los_range})"
        )
    j = summary.junction
    print(rule)
    print(
        f"{'junction':<9}{_fmt(j.control_delay_s) + ' ':>25}{_fmt(j.delay_s) + ' ':>22}"
        f"{'':>31}{_fmt(j.throughput_vph, 0):>20}   {j.los}"
    )
    if summary.baseline:
        moves = baseline_by_movement(summary.baseline)
        text = ", ".join(f"{m} {v:.1f}" for m, v in moves.items())
        print(
            f"\nControl delay = total delay - free-flow baseline ({text} s/veh); "
            "LOS uses control delay."
        )

    notes = [
        f"{a.arm}: LOS F because {', '.join(a.f_reasons)}"
        for a in summary.approaches.values()
        if a.f_reasons
    ]
    if any(a.delay_lower_bound for a in summary.approaches.values()):
        notes.append("* lower bound: some measured vehicles had not finished at the time cap")
    notes += summary.warnings
    if notes:
        print()
    for line in notes:
        print(f"  {line}")


def cmd_run(args: argparse.Namespace) -> int:
    scenario = _load(args.target)
    if args.confirm_demand:
        scenario = scenario.confirm()
    _describe(scenario)
    if scenario.needs_confirmation:
        _print_unconfirmed(scenario)
        return 4
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    result = run_scenario(scenario, run_dir=args.out, seeds=seeds)

    print(f"\nRun directory: {result.dir}")
    print(f"{'seed':>4}  {'ended by':<24}{'sim end':>8}  warnings")
    for r in result.seeds:
        print(f"{r.seed:>4}  {r.ended_by:<24}{r.end_time_s:>7.0f}s  {'; '.join(r.warnings) or '-'}")
    print_summary(result.summary)

    any_warning = result.summary.warnings or any(r.warnings for r in result.seeds)
    return 1 if any_warning and args.strict else 0


def print_webster(report, cfg) -> None:
    sat = report.saturation
    print("Saturation flow and lost time (saturated single-lane run, all seeds)")
    print(
        f"  saturation headway {sat.headway_s:.3f} s -> saturation flow "
        f"{sat.saturation_flow_vph:.0f} veh/h "
        f"(seeds {min(sat.by_seed_vph):.0f}-{max(sat.by_seed_vph):.0f})"
    )
    print(
        f"    from {sat.headways} headways, 5th queued vehicle onwards, "
        f"in {sat.cycles} saturated cycles"
    )
    print(
        f"  vehicles per cycle {sat.vehicles_per_cycle:.2f} -> lost time "
        f"{sat.lost_time_s:.2f} s = green {report.green_s:g} + amber "
        f"{cfg.control.amber_s:g} - {sat.vehicles_per_cycle:.2f} x {sat.headway_s:.3f}"
    )
    print(
        f"  signal: cycle {report.cycle_s:g} s, green {report.green_s:g} s, effective green "
        f"{report.effective_green_s:.2f} s"
    )
    for name, used, measured in (
        ("saturation_flow_vphpl", cfg.control.saturation_flow_vphpl, sat.saturation_flow_vph),
        ("lost_time_s", cfg.control.lost_time_s, sat.lost_time_s),
    ):
        if abs(used - measured) > 0.02 * measured:
            print(f"  NOTE: config.yaml {name} = {used:g} differs from this measurement")

    print(
        f"\nControl delay per vehicle (s), {len(sat.by_seed_vph)} seeds; pass if within "
        f"+-{report.margin_relative:.0%} or +-{report.margin_absolute_s:g} s, whichever is larger"
    )
    header = (
        f"{'x':>5}{'q veh/h':>9}{'green ratio':>13}{'Webster':>9}"
        f"{'simulated control (min-max)':>29}{'total':>7}{'diff s':>8}{'diff %':>8}   result"
    )
    print(header)
    print("-" * (len(header) + 3))
    for c in report.cases:
        result = ("pass" if c.passed else "FAIL") if c.tested else "info only"
        sim = _fmt(c.simulated_s)
        print(
            f"{c.x:>5g}{c.q_vph:>9.0f}{c.green_ratio:>13.3f}{c.webster_s:>9.1f}{sim:>29}"
            f"{c.simulated_total_s.mean:>7.1f}{c.diff_s:>+8.1f}{c.diff_pct:>+7.0f}%   {result}"
        )
    print(
        f"\nFree-flow baseline (no signal, {report.free_flow_q_vph:.0f} veh/h): "
        f"{_fmt(report.free_flow_delay_s, 2)} s/veh; control delay = total - baseline mean"
    )
    print("Overall: " + ("PASS" if report.all_passed else "FAIL"))


def cmd_webster(args: argparse.Namespace) -> int:
    from junctionlab.config import get_config
    from junctionlab.webster import run_check

    cfg = get_config()
    report = run_check(args.out, cfg)
    print_webster(report, cfg)
    return 0 if report.all_passed else 1


def cmd_interpret(args: argparse.Namespace) -> int:
    from junctionlab import llm

    text = args.text
    if text == "-":
        text = sys.stdin.read()
    transport = llm.RecordingTransport(args.record, text) if args.record else None
    try:
        client = llm.make_client(transport=transport)
        result = llm.interpret(text, llm.LlmSession(), client=client)
    except llm.InterpretationError as e:
        print(e)
        print(llm.REPHRASE_HINT)
        return 2
    except llm.LlmError as e:
        print(e)
        return 3
    finally:
        if args.record and args.record.is_file():
            print(f"Recorded response: {args.record}")

    s = result.scenario
    _describe(s)
    for path in s.assumed:
        print(f"    assumed: {path}")
    for path, note in s.interpreted.items():
        print(f"    interpreted: {path}: {note}")
    for note in result.notes:
        print(f"  note: {note}")
    for item in result.not_simulated:
        print(f"  not simulated: {item}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(s.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"Scenario written to {args.out}")
    if args.run:
        if args.confirm_demand:
            s = s.confirm()
        try:
            s.require_confirmed()
        except UnconfirmedDemand:
            _print_unconfirmed(s)
            return 4
        result_run = run_scenario(s)
        print(f"\nRun directory: {result_run.dir}")
        print_summary(result_run.summary)
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    from junctionlab import agent, llm

    scenario = _load(args.target)
    if args.confirm_demand:
        scenario = scenario.confirm()
    if scenario.needs_confirmation:
        _print_unconfirmed(scenario)
        return 4
    try:
        client = llm.make_client()
        goal = agent.read_goal(args.goal, client)
        print(f"Goal read as: {goal.label}")
        print("Working: each simulation takes a few seconds.", flush=True)
        result = agent.run_agent(scenario, goal, client)
    except (llm.LlmError, ValueError) as e:
        print(e)
        return 3
    for step in result.trace:
        print(f"  {_trace_line(step)}")
    print(f"\n{agent.STOP_LABELS[result.stop_reason]} {result.message}".rstrip())
    print(f"Best design ({result.best_id}): {agent.design_label(result.best)}")
    print(f"Goal: {agent.STATUS_LABELS[result.status]}")
    print(f"Simulation runs: {result.runs_used}; LLM calls: {result.llm_calls}")
    print_summary(result.best_summary)
    return 0


def _trace_line(step) -> str:
    return f"{step.step}. {step.change} | {step.result or 'proposed'} | {step.reason}"


def cmd_agent_eval(args: argparse.Namespace) -> int:
    from junctionlab import agent_eval as ev
    from junctionlab import llm

    try:
        if args.stage == "grid":
            print(f"Grid: {ev.run_grid()} designs simulated or already saved.")
        elif args.stage == "baselines":
            ev.run_baselines()
        elif args.stage == "agent":
            print(f"Agent runs completed now: {ev.run_agent_stage()}")
        else:
            print(f"Report written: {ev.write_report()}")
    except (llm.LlmError, ValueError) as e:
        print(e)
        return 3
    return 0


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
    run.add_argument(
        "--confirm-demand", action="store_true", help="accept the model's reading of the demand"
    )
    run.set_defaults(func=cmd_run)
    web = sub.add_parser("webster", help="check simulated delay against Webster's formula")
    web.add_argument("--out", type=Path, help="output directory (default: runs/<time>_webster)")
    web.set_defaults(func=cmd_webster)
    text = sub.add_parser("interpret", help="turn a description into a scenario (uses the API)")
    text.add_argument("text", help="junction description in quotes, or - to read stdin")
    text.add_argument("--out", type=Path, help="write the scenario JSON here")
    text.add_argument("--run", action="store_true", help="simulate the scenario afterwards")
    text.add_argument("--record", type=Path, help="save the API response for replay in tests")
    text.add_argument(
        "--confirm-demand", action="store_true", help="accept the model's reading of the demand"
    )
    text.set_defaults(func=cmd_interpret)
    design = sub.add_parser("agent", help="design a signal for a goal (uses the API)")
    design.add_argument("target", help="example key or path to a JSON file")
    design.add_argument("goal", help="the goal in quotes")
    design.add_argument(
        "--confirm-demand", action="store_true", help="accept the model's reading of the demand"
    )
    design.set_defaults(func=cmd_agent)
    evaluate = sub.add_parser("agent-eval", help="evaluate the agent against baselines")
    evaluate.add_argument("stage", choices=["grid", "baselines", "agent", "report"])
    evaluate.set_defaults(func=cmd_agent_eval)
    args = parser.parse_args(argv)
    # Model notes can hold characters a Windows console cannot encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

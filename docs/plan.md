# JunctionLab – Plan

Scope: [product-brief.md](product-brief.md). Method: [method.md](method.md).

## Milestones

| # | Milestone | Content | Done when |
| --- | --- | --- | --- |
| M0 | Repo skeleton | `pyproject.toml`, `junctionlab/` package, `config.yaml`, pytest, ruff, GitHub Actions on Windows and Ubuntu | CI is green on both systems, including a test that runs `sumo` from the pip package |
| M1 | Scenario model | Pydantic `Scenario` (method §1), defaults from config, tracking of assumed fields | Validation tests pass for all rules in method §1 |
| M2 | SUMO generation | Node, edge, connection, signal and route files; `netconvert`; 5-seed headless runs; time cap; no teleports; gridlock warning | Golden-file tests pass; every control type smoke-runs |
| M3 | Measures | tripinfo and E2 parsing, depart-time window, unfinished vehicles, per-approach mean and range, both LOS tables, overload rules | Unit tests with recorded SUMO output pass |
| M4 | Webster check | Saturation flow and lost time measured from a saturated run; x = 0.5, 0.7, 0.85 (+0.95 for information) | Report produced; margin test passes or the gap is explained (no tuning) |
| M5 | Text to scenario | Claude Haiku 5.5 with structured output into the `Scenario` schema; rephrase on failure; call limits | Tests on recorded responses pass, with no live calls in CI |
| M6 | Streamlit app | Text → editable form (assumed fields highlighted) → run → results table and Plotly charts → sumolib network drawing; copy and compare | Manual check plus app smoke test |
| M7 | PDF report | One page with fpdf2 and matplotlib charts | PDF generated in tests |
| M8 | Examples and docs | 3 bundled scenarios that work offline, README, deployment to Streamlit Community Cloud | Examples run in CI without an API key |

## Decisions

Recorded 2026-11-18, from the question round on the original plan.

1. **Approach length.** 300 m, configurable. Warn when a queue reaches 80% of the approach length.
2. **Bus lane.** Buses only, going straight or right. Separate bus demand in buses/h per approach, default 0. The bus lane is the rightmost lane.
3. **Phase plans.**
   - 4-arm: 2 phases (north–south then east–west, permitted lefts) by default; 4-phase split (one approach at a time) as an option.
   - T-junction: 2 phases, major road then minor road.
   - No protected lefts in v1 (roadmap).
   - 3 s amber + 2 s all-red, configurable.
4. **Green splits.** When not stated, derive them with Webster's method from critical flow ratios, and mark them as assumed in the form.
5. **Arrivals.** Poisson.
6. **Delay.** Delay = `timeLoss + departDelay`, so insertion delay in overloaded cases is counted. Exit edges flow freely, so this is close to control delay (stated in method.md). The Webster cases report the free-flow timeLoss to show it is close to zero.
7. **Teleports.** Disabled. A gridlock warning is added, and every run ends at a time cap.
8. **Queue.** E2 lane-area detectors on every approach lane (`maxJamLengthInMeters`, `maxJamLengthInVehicles`). Per approach: the maximum over the run of the longest lane, not the sum.
9. **LOS.**
   - Both HCM tables (signalised and unsignalised), chosen by control type.
   - Grade F on signal approaches where v/c > 1, using the measured saturation flow.
   - Elsewhere, grade F when throughput falls clearly below demand or the insertion backlog keeps growing.
   - HBS deferred.
10. **Webster check.**
    - Single through lane; saturation flow and lost time measured from a saturated SUMO run.
    - x = 0.5, 0.7, 0.85, plus 0.95 for information only.
    - Margin: ±15% or ±3 s, whichever is larger.
    - SUMO parameters are not tuned to hit the margin; results are reported honestly.
11. **Control types.**
    - Signals at full fidelity.
    - Priority rule basic but correct, with a "major road" field (default north–south; on a T-junction the stem is always minor).
    - All-way stop basic.
12. **Heavy vehicles.** SUMO default truck type; its parameters are listed in method.md.
13. **Measurement window.** Selected by depart time: vehicles that enter during the 60-minute period are measured. The run continues up to 15 minutes more so they can leave, and unfinished vehicles are reported.
14. **Defaults.** `config.yaml` in the repo. The app form edits the scenario, not the config.
15. **Junction drawing.** Render the real SUMO network geometry with sumolib.
16. **PDF charts.** matplotlib.
17. **LLM limit.** 5 calls per session and 50 per day globally, with the counter in a local file. It resets on server restart, which is acceptable; the API spending limit is the hard limit.
18. **Model.** Claude Haiku 5.5, with the model ID in config. Check the current Anthropic docs for the structured-output syntax before writing the LLM code (M5).
19. **Deployment.** Streamlit Community Cloud. The API key is read from Streamlit secrets, with `.env` as the local fallback.
20. **Bundled examples.**
    - A 4-arm fixed-time junction.
    - A T-junction with priority rule.
    - A 4-arm junction with a west bus lane.

**Further decisions**

- Exit legs get the same lane count as the inbound side of that arm.
- All lanes run the full approach length (no short turn bays).
- Krauss driver model with SUMO defaults (sigma 0.5), set in config.
- CI runs on Windows and Ubuntu.

## Roadmap (after v1)

- Protected left-turn phases
- HBS level-of-service thresholds
- Items listed as out of scope in the product brief

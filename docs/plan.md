# JunctionLab – Plan

Scope: [product-brief.md](product-brief.md). Method: [method.md](method.md).

## Working rule

At the end of every milestone: commit, push, check that CI passes on Ubuntu and Windows (fix and push again if it fails), then stop. No need to ask first.

## Milestones

| # | Milestone | Content | Done when |
| --- | --- | --- | --- |
| M0 | Repo skeleton | `pyproject.toml`, `junctionlab/` package, `config.yaml`, pytest, ruff, GitHub Actions on Windows and Ubuntu | CI is green on both systems, including a test that runs `sumo` from the pip package |
| M1 | Scenario model | Pydantic `Scenario` (method §1), defaults from config, tracking of assumed fields | Validation tests pass for all rules in method §1 |
| M2 | SUMO generation | Node, edge, connection, signal and route files; `netconvert`; 5-seed headless runs; time cap; no teleports; gridlock warning | Golden-file tests pass; every control type smoke-runs |
| M3 | Measures | tripinfo and E2 parsing, depart-time window, unfinished vehicles, per-approach mean and range, both LOS tables, overload rules | Unit tests with recorded SUMO output pass |
| M4 | Webster check | Saturation flow and lost time measured from a saturated run; x = 0.5, 0.7, 0.85 (+0.95 for information) | Report produced; margin test passes or the gap is explained (no tuning) |
| M5 | Text to scenario | Configurable LLM (Groq by default, Anthropic optional) with structured output into the `Scenario` schema; rephrase on failure; call limits | Tests on recorded responses pass, with no live calls in CI |
| M6 | Streamlit app | Text → editable form (assumed and interpreted fields highlighted; interpreted demand shown with the model's note and confirmed before Run is enabled) → run → results table and Plotly charts → sumolib network drawing; copy and compare | Manual check plus app smoke test |
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
18. **Model.** Originally Claude Haiku 5.5. Replaced as the default by Groq on 2026-11-22 (see "LLM provider" below). Model IDs are in config.
19. **Deployment.** Streamlit Community Cloud. The provider's API key is read from Streamlit secrets, with `.env` as the local fallback.
20. **Bundled examples.**
    - A 4-arm fixed-time junction.
    - A T-junction with priority rule.
    - A 4-arm junction with a west bus lane.

**Further decisions**

- Exit legs get the same lane count as the inbound side of that arm.
- All lanes run the full approach length (no short turn bays).
- Krauss driver model with SUMO defaults (sigma 0.5), set in config.
- CI runs on Windows and Ubuntu.

**Later decisions**

- **T-junction example volumes:** main road 350 + 350 veh/h, stem 140 veh/h, which puts the stem at LOS C–D (2026-11-18, M2). The first volumes (700 + 650, stem 250) overloaded the stem and are kept as a test case for the time cap and gridlock warning (`tests/data/t_junction_overloaded.json`).
- **Permitted lefts in example 1 stay as they are,** even though east and west lefts in the shared lanes see high delays.
- **Comparison demo:** example 1 run as two-phase vs four-phase split.
- **Measured values replace the placeholder (M4).** Saturation flow is 1,861 veh/h per lane, replacing 1,800. Lost time is 5.75 s per phase. Both now feed the Webster green splits (on effective green) and the signal v/c.
  - Effect on example 1: the greens stay 49 / 31 s, so delays, queues and LOS are unchanged. v/c rises slightly: north 0.408 → 0.418, east and west 0.403 → 0.428, south 0.306 → 0.314.
- **Webster test signal:** 90 s cycle with 40 s green, matching a two-phase junction with equal greens (M4).
- **Control delay (after M4).** LOS and the Webster comparison use control delay: total delay minus a free-flow baseline per approach and movement. The baseline comes from a low-demand run with the junction unregulated. Total delay stays in `summary.json` and in the output.
  - Webster: the baseline is 2.1 s, and the differences are now −0.6, −0.9 and −3.0 s (all pass); x = 0.95 is −23.1 s.
  - Example 1: east and west change from LOS D to C (34.1 and 34.5 s, just under the 35 s boundary). North's seed range widens to B–C.
- **LLM call (M5).**
  - Messages API `create` with `output_config.format` (JSON schema built by `anthropic.transform_schema`), not `messages.parse`. That way the stop reason is checked before parsing, and a refused or cut-off answer is not reported as a schema error.
  - Effort `medium`, the Haiku 5.5 default, set in config.
  - These settings still apply to the optional Anthropic path. Its fixtures in `tests/data/llm_anthropic/` are synthetic, because no Anthropic key was used.
- **LLM provider (2026-11-22).** No paid API: `llm.provider` in `config.yaml` is `groq` (default) or `anthropic`.
  - Groq model: `openai/gpt-oss-120b`, a production model that supports strict structured outputs. The request uses the same JSON schema as the Anthropic path.
  - A 429 is not retried. The user is told how long to wait, using `retry-after`.
  - `tests/data/llm/` now holds real Groq responses (`"source": "live"`), recorded with `record_groq.py`. The 429 and refusal cases are constructed in the tests, because they cannot be triggered on demand.
  - Real responses showed that the model listed optional details as `unclear`, which rejected a valid example. The prompt now limits `unclear` to the required facts.
  - A stated detail such as a pedestrian crossing can still be dropped from the notes.
- **Field states (2026-11-22).** Every scenario field is `stated`, `interpreted` (the text allowed several readings and the model chose one) or `assumed` (a default).
  - The model must report its readings in `interpreted`. Code also marks demand as interpreted when the text gives no time period, because the model missed that case in testing.
  - Interpreted demand blocks the run until it is confirmed. `run_scenario` enforces this, so the M6 Run button cannot bypass it. M6 must show the note and call `Scenario.confirm()`. Editing a demand value makes it stated.
  - On the command line, `--confirm-demand` accepts the reading.
- **Backlog rule threshold:** "the insertion backlog keeps growing" means it grows by at least 5 vehicles from the start to the end of the measurement period. Confirmed after M3.
- **Throughput is counted at stop-line E1 loops,** not from tripinfo, because tripinfo cannot show when a vehicle crossed the junction (M3).

## Roadmap (after v1)

- Protected left-turn phases
- HBS level-of-service thresholds
- Items listed as out of scope in the product brief

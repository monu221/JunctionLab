# JunctionLab

JunctionLab turns a plain-language description of a road junction into a SUMO traffic simulation and reports delay, queues and level of service (LOS) per approach.

![JunctionLab results for the 4-arm signal example: junction LOS, control delay per approach and the junction drawn from the SUMO network](docs/screenshot.png)

**Live demo:** _link to be added after deployment to Streamlit Community Cloud_

## What it does

- **Text to scenario.** An LLM reads a description such as "four-arm junction, two lanes per approach, 800 veh/h from the north, fixed-time signal with a 90 s cycle" into a structured scenario. It never writes SUMO files. The scenario appears as an editable form in which every field is marked:
  - **stated:** taken from the text, or edited by you
  - **interpreted:** the model chose one reading, for example of counts given with no time period. You confirm it before Run is enabled.
  - **assumed:** a default from `config.yaml`
- **Simulation.** Python generates the network, routes and signal plan. SUMO then runs 5 seeds headless, with a warm-up, a one-hour measurement period and no teleports. It covers fixed-time signals, priority rules and all-way stops on 4-arm and T-junctions, with heavy vehicles and bus lanes.
- **Results per approach.** Control delay, total delay, maximum queue, throughput and LOS are shown as mean and range over the seeds. The junction is drawn from the SUMO network, with its lanes coloured by LOS.
- **Comparison.** Keep a run as A, edit the scenario, run it again, and compare the two approach by approach.
- **Signal-design agent.** Give a goal such as "all approaches LOS C or better, shortest cycle possible". A tool-using agent then changes the cycle, the green splits and the phase plan, and checks each design in SUMO. It cannot change demand or geometry. Python ranks the designs, not the model. You see the trace and the best design with a before/after table, and can adopt it into the form.

Three bundled examples (4-arm signal, T-junction with a priority rule, 4-arm with a bus lane) run with no API call.

## Method

The full method is in [docs/method.md](docs/method.md), and every default is in `config.yaml`.

**Delay.** Total delay per vehicle is SUMO's `timeLoss + departDelay`. Adding `departDelay` means queues that back up beyond the network edge are still counted. **Control delay** is total delay minus a free-flow baseline. That baseline is measured on the same network, made unregulated, at 60 cars/h per movement, for each approach and movement. It is about 2–5 s per vehicle. LOS is graded on control delay, as in the HCM.

**LOS thresholds** (average control delay, s/veh, US Highway Capacity Manual):

| LOS | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- |
| Signalised | ≤ 10 | 10–20 | 20–35 | 35–55 | 55–80 | > 80 |
| Priority rule / all-way stop | ≤ 10 | 10–15 | 15–25 | 25–35 | 35–50 | > 50 |

An approach is graded F regardless of delay when it is overloaded. For signals, that is v/c > 1. For priority and all-way-stop approaches, it is throughput below 95% of demand, or an insertion backlog that grows by 5 or more vehicles.

**Webster check.** A single fixed-time approach (90 s cycle, 40 s green) is simulated and compared with Webster's delay formula. Saturation flow (1,861 veh/h) and lost time (5.75 s) are measured from a saturated SUMO run, not assumed. Results from 2026-11-18 (SUMO 1.28, 5 seeds):

| x | Webster (s) | Simulated control delay, mean (range) (s) | Difference | Result |
| --- | --- | --- | --- | --- |
| 0.50 | 21.1 | 20.5 (18.2–22.9) | −0.6 s (−3%) | pass |
| 0.70 | 24.8 | 23.9 (22.9–25.5) | −0.9 s (−3%) | pass |
| 0.85 | 32.4 | 29.4 (24.8–34.5) | −3.0 s (−9%) | pass |
| 0.95 | 63.0 | 40.0 (30.3–46.0) | −23.1 s (−37%) | information only |

The pass margin is ±15% or ±3 s, whichever is larger. No SUMO parameter was tuned to meet it. At x = 0.95 Webster assumes a steady-state queue that a one-hour period does not reach, as explained in method §4. Run the check with `python -m junctionlab webster`.

## Agent evaluation

The agent was compared with Webster's optimum cycle, random search and a grid search. The test cases were two junctions (the 4-arm example and a signalised copy of the T-junction) and three goals, and every method that involves chance ran 3 times. Each method chose its design on seeds 1–5 and was judged on unused seeds 6–10. Full results are in [docs/agent_evaluation.md](docs/agent_evaluation.md).

Results on the unused seeds. For the agent and random search, each value is the median of 3 repeats.

| Junction and goal | Agent | Webster | Random search | Grid optimum |
| --- | --- | --- | --- | --- |
| 4-arm, LOS C with shortest cycle | 60 s | 43 s | 42 s | 40 s |
| 4-arm, lowest junction delay | 21.9 s/veh | 20.1 s/veh | 25.8 s/veh | 19.2 s/veh |
| 4-arm, lowest worst-approach delay | 26.3 s/veh | 24.4 s/veh | 27.7 s/veh | 22.6 s/veh |
| T-junction, LOS C with shortest cycle | 72 s | 40 s | 50 s | 40 s |
| T-junction, lowest junction delay | 13.3 s/veh | 14.0 s/veh | 15.0 s/veh | 11.2 s/veh |
| T-junction, lowest worst-approach delay | 25.5 s/veh | 19.5 s/veh | 31.1 s/veh | 17.9 s/veh |
| Simulation runs per design | about 8 | 1 | 8 | 150–170 |

- **Agent against Webster.** The agent beat Webster's design on one goal of six, the T-junction's lowest junction delay. On the other five goals Webster's single calculation was better, and the grid optimum was best on every goal.
- **Agent against random search.** The agent was more reliable with the same budget: 17 of its 18 runs met the goal and 1 was borderline. Random search met it in 14 runs, was borderline in 1 and failed in 3, with one failed run reaching 209 s/veh.
- **Weakest goal.** The shortest-cycle goal is the agent's weakest: it stopped at cycles of 60–75 s where 40–43 s already meets LOS C.

## Limitations

- One junction at a time, cars, heavy vehicles and buses only. Pedestrians, cyclists, adaptive signals, protected left-turn phases and imports from map data are not modelled. The app lists anything in a description that it does not simulate.
- LOS uses HCM thresholds. The German HBS thresholds are not included yet.
- Delay close to capacity is uncertain: at x = 0.95, Webster's formula and the simulation differ by 37%. Delays of vehicles still in the network at the time cap are lower bounds, and they are flagged as such.
- The LLM can misread a description, which is why every field is marked and editable, and an interpreted demand must be confirmed.
- **Free Groq tier.** Reading descriptions and the agent run on Groq's free tier (`openai/gpt-oss-120b`). The app allows 5 description reads per session and 50 per day, and the agent 3 goals per session and 150 LLM calls per day. Groq's own per-minute and per-day limits also apply. When a limit is reached, the app says so, and the examples still work. To use Anthropic instead (a paid API), set `llm.provider: anthropic` in `config.yaml` and add `ANTHROPIC_API_KEY`.

## Run locally on Windows

You need Python 3.12. SUMO comes from the `eclipse-sumo` pip package, so no separate install is needed.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env          # then add your GROQ_API_KEY (free at console.groq.com)
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Without a key, the examples and the form still work. On Linux or macOS, activate with `source .venv/bin/activate`.

The command line:

```sh
python -m junctionlab list                      # bundled examples
python -m junctionlab run four_arm_signal       # simulate 5 seeds, print delay/queue/throughput/LOS
python -m junctionlab run my_scenario.json --seeds 1,2 --out runs/test
python -m junctionlab webster                   # the Webster check above
python -m junctionlab interpret "Crossroads, 800 veh/h from the north, ..." --out s.json --run
python -m junctionlab agent four_arm_signal "All approaches LOS C or better, shortest cycle possible"
```

If the model had to choose a reading of the demand, `--run` stops and shows its note. Correct the description, or add `--confirm-demand` to accept that reading.

## Deploy to Streamlit Community Cloud

- **Main file path:** `app/streamlit_app.py`. Choose Python 3.12 under Advanced settings.
- `requirements.txt` installs this package, and `packages.txt` adds the X11 and OpenGL libraries that the SUMO binary in the `eclipse-sumo` Linux wheel links against.
- **Secrets.** Paste the contents of [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example) into the app's Secrets, with your key:
  - `GROQ_API_KEY`: needed to read descriptions and run the agent.
  - `JUNCTIONLAB_MAX_PARALLEL_RUNS = 2`: Community Cloud has 2 cores, so a run starts at most 2 SUMO processes at once. Locally, `config.yaml` runs all 5 seeds at once.
- The daily LLM call counters are kept in `var/`, which Community Cloud clears when the app restarts.

## Tests

```sh
ruff check . && ruff format --check .
pytest
```

The tests run real SUMO but make no API calls: LLM responses are recorded. CI runs them on Ubuntu and Windows. To run the one live test:

```sh
JUNCTIONLAB_LIVE_LLM=1 pytest tests/test_llm.py -k live
```

## Roadmap

- **Agent seeded with Webster's design.** Start the agent from Webster's cycle and splits instead of the current plan. In the evaluation, Webster's single calculation beat the agent on five of six goals.
- One-page PDF report
- Protected left-turn phases
- HBS level-of-service thresholds
- Later: pedestrians and cyclists, networks of several junctions, adaptive signals, import from map data

## Licence

MIT, see [LICENSE](LICENSE).

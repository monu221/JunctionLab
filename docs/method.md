# JunctionLab – Method

Decisions behind this method are recorded in [plan.md](plan.md#decisions).

## 1. Text to scenario

The LLM returns a scenario in a fixed JSON schema. Python validates it with Pydantic before anything runs.

| Field | Example | Rules |
| --- | --- | --- |
| Junction type | 4-arm or 3-arm (T) | v1 supports these two only |
| Approaches | north, east, south, west | One entry per arm |
| Lanes per approach | 2, with lane use (left, through, right) | 1 to 4 lanes. The exit leg of an arm has the same lane count as its inbound side. |
| Approach length | 300 m | Configurable. Every lane runs the full approach length (no short turn bays). |
| Bus lane | yes on west approach | Optional, per approach. Always the rightmost lane, buses only; buses go straight or right. |
| Bus demand | 12 buses/h from west | Per approach, buses per hour, default 0. Separate from vehicle demand. |
| Demand | 800 veh/h from north | Per approach, vehicles per hour |
| Turning shares | 20% left, 65% through, 15% right | Must sum to 100% per approach |
| Heavy vehicle share | 5% | 0 to 30% |
| Control | Fixed-time signal, priority rule, or all-way stop | See below |
| Duration | 10 min warm-up, 60 min measured, up to 15 min run-out | See section 2 |

**Fixed-time signal.** Needs a cycle length. Phase plans:

- 4-arm, default: 2 phases, north–south then east–west, with permitted (not protected) lefts.
- 4-arm, option: 4-phase split, one approach at a time.
- T-junction: 2 phases, major road then minor road.
- Protected left phases are not in v1 (roadmap).
- Intergreen: 3 s amber + 2 s all-red per phase change. Configurable.
- If the text gives no green splits, they come from Webster's method, splitting effective green in proportion to the critical flow ratio of each phase. They are marked as assumed.

**Priority rule.** A "major road" field sets which road has priority (default north–south). On a T-junction the stem is always the minor road. This is a basic model, but it must be correct: minor-road traffic yields to all conflicting major-road movements.

**All-way stop.** Basic model using SUMO's `allway_stop` junction type.

**Defaults.** These are used when the text does not state a value, and the form marks them as assumed:

| Item | Default |
| --- | --- |
| Speed limit | 50 km/h (German urban default) |
| Approach length | 300 m |
| Lanes per approach | 2: left + through, and through + right |
| Heavy vehicle share | 5% |
| Bus demand | 0 buses/h |
| Cycle length | 90 s |
| Minimum green | 5 s |
| Intergreen | 3 s amber + 2 s all-red |
| Green splits | Webster, from critical flow ratios |
| Saturation flow (for green splits) | 1,800 veh/h per lane: placeholder, replaced by the measured value in M4 |
| 4-arm phase plan | 2-phase |
| Major road (priority rule) | north–south |
| Lane use from a lane count | Leftmost lane left + through, rightmost lane through + right, middle lanes through. A single lane takes all movements. |
| Turning shares, 4-arm | 15% left, 70% through, 15% right |
| Turning shares, T-junction | Major road: 80% through, 20% into the stem. Stem: 50% left, 50% right |
| Bus direction | All buses straight |

**Required.** Junction type, demand on every approach and control type must come from the text. There is no sensible default for them, so the app asks the user to rephrase.

**Modelling rules.**

- Approaches are named after the arm vehicles come from. Traffic drives on the right.
- Lanes are listed from leftmost to rightmost. Movements must not cross between lanes. Every movement with a share above 0 needs a lane.
- The bus lane is an extra lane to the right of the general-traffic lanes. It counts towards the 4-lane maximum and towards the lane count of the exit leg.
- On a T-junction, the stem is the arm opposite the missing arm. Default lane uses lose movements that would lead to the missing arm.
- No bus lane and no bus demand mean the feature isn't used. They are not marked as assumed.
- **Green splits.** When they are not given, they are derived with Webster's split method and marked as assumed. The steps:
  1. Each movement's flow is split equally over the lanes that serve it.
  2. Each phase's critical flow ratio is its highest lane flow divided by the saturation flow. Heavy vehicles count as one vehicle each while the saturation flow is a placeholder.
  3. The available green (cycle minus one intergreen per phase) is shared in proportion to these ratios. Phases below the minimum green are raised to it, and the result is rounded to whole seconds.

  Editing the scenario in the form re-derives splits that are still assumed.

Anything the text does not state gets a default from `config.yaml`, and the form highlights assumed fields so the user can check them. The form edits the scenario only, never the config. If the text cannot be interpreted, the app asks the user to rephrase instead of guessing.

## 2. Scenario to simulation

Python writes the SUMO node, edge, connection, signal and route files, builds the network with `netconvert`, and runs `sumo` headless.

- **Network.**
  - The junction node sits at the centre, and each arm has an inbound edge and an exit edge with the same lane count.
  - The arm end nodes are placed so that every inbound lane is exactly the approach length long, measured from the edge of the junction area.
  - Lane connections are written explicitly:
    - Left turns use the leftmost exit lanes, and right turns use the rightmost.
    - Through lanes keep their position among the general lanes.
    - The bus lane connects to the rightmost exit lane for through and right.
- **Signal states.** Each phase is green, then amber, then all-red:
  - Links from approaches served by the phase get a major green (`G`).
  - Left turns facing a served opposite approach get a permitted green (`g`) and yield to oncoming traffic.
  - When two `G` links would enter the same exit lane, the lower-ranked one yields (`g`). The ranking is general through, then general right, then bus lane, then left. This applies, for example, where a bus lane merges at the exit.
  - Phases: 4-arm two-phase serves north–south, then east–west. 4-phase split serves one approach at a time. A T-junction serves the main road, then the stem.
- **Priority rule.** The major-road edges get a higher edge priority in `netconvert`, so minor-road traffic yields. An all-way stop uses SUMO's `allway_stop` junction type.

- **Seeds.** Each scenario runs 5 times with different random seeds. Results are reported as mean and range. Comparisons use the same seeds for both scenarios.
- **Arrivals.** Poisson: SUMO flows with random exponential headways (`period="exp(rate)"`), one flow per approach, movement and vehicle type (car, truck, bus). Vehicles that cannot enter wait at the start of the approach, and their wait counts as `departDelay`.
- **Driver model.** Krauss, using the parameters in `config.yaml` (SUMO defaults, sigma 0.5).
- **Vehicle types.** Cars use SUMO's default passenger type. Heavy vehicles use SUMO's default truck type, unchanged. In SUMO 1.28 its values are:

  | Parameter | Truck | Passenger car (for reference) |
  | --- | --- | --- |
  | length (m) | 7.1 | 5.0 |
  | minGap (m) | 2.5 | 2.5 |
  | accel (m/s²) | 1.3 | 2.6 |
  | decel (m/s²) | 4.0 | 4.5 |
  | emergencyDecel (m/s²) | 7.0 | 9.0 |
  | maxSpeed (km/h) | 130 | 200 |
  | sigma | 0.5 | 0.5 |
  | tau (s) | 1.0 | 1.0 |
  | speedDev | 0.05 | 0.1 |
  | width (m) | 2.4 | 1.8 |

  These values were read through TraCI from the pip `eclipse-sumo` 1.28.0 package. Re-check them when the SUMO version changes.

- **Step length.** 1.0 s, the SUMO default. If the Webster check shows problems that a finer step would fix, they are reported before anything is changed.
- **Timing.** The run has three stages:
  1. A 10-minute warm-up.
  2. A 60-minute measurement period.
  3. A run-out of up to 15 minutes, so that measured vehicles can leave.

  Only vehicles that depart during the measurement period are counted, selected by depart time. Measured vehicles still in the network at the end of the run-out are reported as unfinished.

  Demand continues during the run-out, so the last measured vehicles see normal traffic. Run-out vehicles have ids starting with `ro_` and are never measured. A run ends as soon as every warm-up and measurement vehicle has left, or at the time cap at the latest.
- **No teleports.** Teleporting is disabled (`--time-to-teleport -1`). Every run stops at the hard time cap above. If any vehicle has been stopped for longer than 300 s (default), the run reports a gridlock warning instead of silently removing the vehicle. Stopped time is the uninterrupted time below 0.1 m/s, checked through TraCI every 10 s of simulated time.
- **Collisions.** Collisions are logged and counted (`--collision.action warn`), never resolved by removing vehicles. A clean run has none.
- **Detectors.** An E2 lane-area detector covers the full length of every inbound lane and writes 60-second intervals.
- **Spill-back warning.** A warning is raised when the maximum queue on an approach reaches 80% of its length.

## 3. What is measured

| Measure | Definition | SUMO source |
| --- | --- | --- |
| Average delay | `timeLoss + departDelay` per measured vehicle (s/veh) | `tripinfo` output, plus the runner's record of vehicles still waiting to enter |
| Maximum queue | Per approach: the maximum over the measurement period of the longest lane queue (m and vehicles) | E2 lane-area detector on every approach lane: `maxJamLengthInMeters`, `maxJamLengthInVehicles` |
| Throughput | Vehicles passing the junction per hour during the measurement period | E1 loop 1 m before the stop line of every approach lane (`nVehContrib`) |
| Demand | Measured vehicles generated per hour (this seed's Poisson draw, not the nominal value) | `tripinfo` plus vehicles that never entered |
| Unfinished vehicles | Measured vehicles still in the network, or still waiting to enter, at the time cap | `tripinfo` with `--tripinfo-output.write-unfinished`, and `backlog.json` |
| Level of service | Grade A to F from average delay and the overload rules below | Tables below |

**Delay.** `timeLoss` is the time lost compared with driving at the desired speed. `departDelay` is the time a vehicle waited to be inserted. Adding `departDelay` means queues that back up beyond the network edge in overloaded cases are still counted. Exit legs flow freely, so this sum is close to the HCM control delay. It is not identical: it also includes small losses on the approach and exit legs outside the junction influence area. The Webster test cases report the timeLoss of a free-flow run to show that this background loss is close to zero.

**Measured vehicles.** Vehicles from the warm-up and measurement flows whose intended departure (`depart − departDelay`) falls in the measurement period. Run-out vehicles (`ro_`) are never measured.

**Unfinished vehicles.** At the time cap, vehicles still driving count with their delay so far. Vehicles still waiting to enter count with their wait so far, measured from when the runner first saw them waiting (within 10 s). Both make the approach delay a lower bound, which is flagged in the output. SUMO's tripinfo leaves out vehicles that never entered, so the runner records them (`backlog.json`).

**Queue.** For each approach, the reported queue is the longest single lane, not the sum of its lanes. It is the largest value from any 60-second detector interval inside the measurement period. Vehicle counts come directly from the detector. A spill-back warning is raised when the queue reaches 80% of the approach length in any seed.

**Throughput.** Counted at the stop line rather than from tripinfo, so it measures vehicles that actually crossed the junction during the measurement period.

**Level of service.** The table is chosen by control type. Thresholds are average delay per vehicle, as in the US Highway Capacity Manual. They live in `config.yaml`.

| LOS | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- |
| Signalised (s/veh) | ≤ 10 | 10–20 | 20–35 | 35–55 | 55–80 | > 80 |
| Priority rule / all-way stop (s/veh) | ≤ 10 | 10–15 | 15–25 | 25–35 | 35–50 | > 50 |

An approach is graded F regardless of delay when it is overloaded:

- **Signal approaches:** v/c > 1. v is the approach's critical lane flow from nominal demand. Capacity is saturation flow × green / cycle for the phase serving the approach. Until M4, the saturation flow is the 1,800 veh/h placeholder.
- **Priority and all-way-stop approaches:** either of two conditions:
  - Throughput is below 95% of demand (default).
  - The insertion backlog (vehicles waiting to enter) grows by at least 5 vehicles from the start to the end of the measurement period (default).

**Aggregation over seeds.** Every measure is reported per approach as mean, minimum and maximum over the 5 seeds.
- The approach grade is taken from the mean delay. The overload rules use the mean throughput, demand and backlog growth.
- The grades of the individual seeds are shown as a range, for example C–D.
- A junction row gives the vehicle-weighted mean delay and the total throughput. Signalised and all-way-stop junctions also get a junction LOS from that delay. A priority junction gets none, because HCM grades only its minor movements.

German practice uses the HBS (Handbuch für die Bemessung von Straßenverkehrsanlagen), which also grades A to F with its own thresholds. It is deferred: check its thresholds against the current HBS edition before adding them.

## 4. Plausibility check against theory

For a single fixed-time approach clearly below capacity, Webster's formula gives the expected average delay per vehicle:

d = c(1 − λ)² / (2(1 − λx)) + x² / (2q(1 − x)) − 0.65 (c / q²)^(1/3) · x^(2 + 5λ)

- c: cycle length (s)
- λ: effective green ratio (effective green / cycle)
- q: arrival flow (veh/s)
- x: degree of saturation (flow / capacity)

**Setup.**

- A single approach with one through lane, a fixed-time signal, cars only, and Poisson arrivals.
- Saturation flow and lost time are not assumed. They are measured first from a saturated SUMO run on the same geometry: discharge headways after the start-up period give the saturation flow, and the shortfall at the start of green gives the lost time.
- Test cases at x = 0.5, 0.7 and 0.85 must agree with Webster within ±15% or ±3 s, whichever is larger.
- x = 0.95 is reported for information only and is not tested.
- The free-flow timeLoss (green all the time) is reported to show that background loss is close to zero.

Some gap is expected, because SUMO models acceleration, start-up loss and driver behaviour that the formula simplifies. SUMO parameters are not tuned to hit the margin. If a case fails, the gap is reported and explained.

## 5. Comparing scenarios

Two scenarios are compared approach by approach: delay, queue and LOS before and after, with the change highlighted. Both scenarios run with the same 5 seeds.

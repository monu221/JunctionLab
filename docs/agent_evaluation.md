# Signal-design agent: evaluation

<!-- summary -->
## Summary

All 18 agent runs (2 junctions × 3 goals × 3 repeats) are in. All figures below are on the
unused seeds.

- **The agent beat Webster on one goal of six: T-junction G2.** There its median is
  13.3 s/veh against Webster's 14.0, which is still above the grid's 11.2. On all other goals it is
  behind Webster: 21.9 against 20.1 s/veh (4-arm G2), 26.3 against 24.4 (4-arm G3) and
  25.5 against 19.5 (T-junction G3). The grid optimum is best everywhere.
- **G1 (shortest cycle) is its weakest goal.** Its median cycle is 60 s on the 4-arm junction,
  where Webster meets LOS C at 43 s and the grid at 40 s. One of three repeats stopped by itself
  at a borderline 100 s cycle. On the T-junction all three repeats ended at 72–75 s, where 40 s
  (the search-space minimum) already meets the goal.
- **On T-junction G3 it never improved on its first design.** All three repeats kept the
  90 s cycle from run 1 (25.5 s/veh, against Webster's 19.5). One stopped by itself after 3 runs,
  and two used all 15 LLM calls after only 3 and 7 of the 8 allowed simulations.
- **It is more reliable than random search with the same budget.** With about 8 simulation runs
  each, random search failed or was borderline in one of three repeats on four of the six goals
  (for example 209 s/veh in 4-arm G2). Every agent repeat met the goal except one borderline
  4-arm G1 repeat. Its medians are also below random search's on four of the six goals. The
  exceptions are the two G1 goals, where random search found shorter cycles.
- **The LLM-call budget often ends a run before the simulation budget.** This happened in 6 of
  the 18 runs (2 of 3 for 4-arm G3, T-junction G2 and T-junction G3). They had not used all 8
  simulations, so more LLM calls per run may help more than more simulations.
- **The unused seeds confirm the choices.** No method's ranking changes between the seeds used
  to choose a design and the unused ones. Medians differ by at most 2.8 s/veh, and that largest
  change is the agent on T-junction G3, which is lower on the unused seeds.
- Webster's cycle for the T-junction is below the 40 s minimum, so there Webster and the grid
  give the same G1 design by construction.
<!-- /summary -->

Each method chose its best design on seeds [1, 2, 3, 4, 5]. **Unused seeds** re-runs that design on seeds [6, 7, 8, 9, 10], and **Goal** is judged there: met means every approach in every seed, borderline means on the mean only. Repeated methods show the median and range over 3 repeats. Runs are simulation runs used to choose a design.

## Example 1: 4-arm signal

### G1: All approaches LOS C or better, shortest cycle possible.

| Method | Best result | Unused seeds | Goal | Runs |
| --- | --- | --- | --- | --- |
| Agent | 60 s cycle (58–100) | 60 s cycle (58–100) | 2/3 met, 1/3 borderline | 7.3 (best at run 7.3) |
| Webster | 43 s cycle | 43 s cycle | Met | 1 |
| Random search | 42 s cycle (40–75) | 42 s cycle (40–75) | 2/3 met, 1/3 not met | 8 |
| Grid optimum | 40 s cycle | 40 s cycle | Met | 170 |

### G2: Lowest average delay over the junction.

| Method | Best result | Unused seeds | Goal | Runs |
| --- | --- | --- | --- | --- |
| Agent | 22.2 s/veh (21.5–22.7) | 21.9 s/veh (20.6–22.3) | 3/3 met | 8.0 (best at run 6.7) |
| Webster | 20.3 s/veh | 20.1 s/veh | Met | 1 |
| Random search | 25.0 s/veh (20.4–232.4) | 25.8 s/veh (20.0–209.1) | 2/3 met, 1/3 not met | 8 |
| Grid optimum | 19.2 s/veh | 19.2 s/veh | Met | 170 |

### G3: Lowest delay on the worst approach.

| Method | Best result | Unused seeds | Goal | Runs |
| --- | --- | --- | --- | --- |
| Agent | 25.5 s/veh (23.9–30.4) | 26.3 s/veh (23.9–31.9) | 3/3 met | 7.3 (best at run 3.7) |
| Webster | 23.9 s/veh | 24.4 s/veh | Met | 1 |
| Random search | 27.3 s/veh (26.2–608.2) | 27.7 s/veh (27.4–550.7) | 2/3 met, 1/3 not met | 8 |
| Grid optimum | 21.0 s/veh | 22.6 s/veh | Met | 170 |

## T-junction, signalised copy

### G1: All approaches LOS C or better, shortest cycle possible.

| Method | Best result | Unused seeds | Goal | Runs |
| --- | --- | --- | --- | --- |
| Agent | 72 s cycle (72–75) | 72 s cycle (72–75) | 3/3 met | 8.0 (best at run 7.0) |
| Webster | 40 s cycle | 40 s cycle | Met | 1 |
| Random search | 50 s cycle (48–102) | 50 s cycle (48–102) | 2/3 met, 1/3 borderline | 8 |
| Grid optimum | 40 s cycle | 40 s cycle | Met | 150 |

### G2: Lowest average delay over the junction.

| Method | Best result | Unused seeds | Goal | Runs |
| --- | --- | --- | --- | --- |
| Agent | 14.6 s/veh (14.2–15.4) | 13.3 s/veh (13.0–14.8) | 3/3 met | 7.3 (best at run 5.3) |
| Webster | 14.9 s/veh | 14.0 s/veh | Met | 1 |
| Random search | 15.8 s/veh (14.9–17.0) | 15.0 s/veh (12.7–16.1) | 3/3 met | 8 |
| Grid optimum | 12.8 s/veh | 11.2 s/veh | Met | 150 |

### G3: Lowest delay on the worst approach.

| Method | Best result | Unused seeds | Goal | Runs |
| --- | --- | --- | --- | --- |
| Agent | 28.3 s/veh | 25.5 s/veh | 3/3 met | 4.3 (best at run 1.0) |
| Webster | 21.4 s/veh | 19.5 s/veh | Met | 1 |
| Random search | 31.6 s/veh (22.5–36.6) | 31.1 s/veh (20.9–34.0) | 3/3 met | 8 |
| Grid optimum | 17.6 s/veh | 17.9 s/veh | Met | 150 |

## Note on the free tier

A 429 from Groq's free tier ended 9 agent run(s) early (4 of them by the daily token quota; 5 were recorded before the kind of limit was saved, so for those it is not known). Those runs were set aside and run again from the start; they are not in the tables above. None of the runs in the tables needed a wait for a per-minute limit. A wait, when one happens, does not change what the model sees or how many calls it gets. The daily quota is not waited out: the stage stops and resumes on a later day.

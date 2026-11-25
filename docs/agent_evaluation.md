# Signal-design agent: evaluation

<!-- summary -->
## Summary (preliminary, 10 of 18 agent runs)

These conclusions may change once the remaining 8 agent runs (T-junction G1 #2 and #3, G2 and
G3) are in. All figures below are on the unused seeds.

- **The agent did not beat Webster on any goal so far.** On the 4-arm junction its median is
  21.9 s/veh for G2 against Webster's 20.1 and the grid's 19.2, and 26.3 s/veh for G3 against
  24.4 and 22.6. For G1 its median cycle is 60 s, where Webster meets LOS C at 43 s and the
  grid at 40 s.
- **G1 (shortest cycle) is its weakest goal.** On the 4-arm junction one of three repeats
  stopped by itself at a 100 s cycle that is only borderline. On the T-junction, the one
  repeat so far ended at 75 s, where 40 s (the search-space minimum) already meets the goal.
- **It is more reliable than random search with the same budget.** With about 8 simulation
  runs each, random search failed the goal in one of three repeats on every 4-arm goal
  (for example 209 s/veh in G2), while every agent repeat met the goal except one G1 repeat
  that was borderline.
- **The budget rule that ends a run differs by goal.** Two of the three 4-arm G3 runs used all
  15 LLM calls after 7 of the 8 allowed simulations; their best designs were found at runs
  2 and 3.
- **The unused seeds confirm the choices.** No method's ranking changes between the seeds
  used to choose a design and the unused ones; medians differ by at most about 2 s/veh.
- Webster's cycle for the T-junction is below the 40 s minimum, so there Webster and the grid
  give the same G1 design by construction.
<!-- /summary -->

> **Preliminary, 10 of 18 agent runs.** The other agent runs wait for the free tier's daily quota. Webster, random search and the grid are complete.

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
| Agent | 75 s cycle | 75 s cycle | Met | 8.0 (best at run 6.0); 1 of 3 repeats |
| Webster | 40 s cycle | 40 s cycle | Met | 1 |
| Random search | 50 s cycle (48–102) | 50 s cycle (48–102) | 2/3 met, 1/3 borderline | 8 |
| Grid optimum | 40 s cycle | 40 s cycle | Met | 150 |

### G2: Lowest average delay over the junction.

| Method | Best result | Unused seeds | Goal | Runs |
| --- | --- | --- | --- | --- |
| Agent | not run yet | – | – | 0 of 3 repeats |
| Webster | 14.9 s/veh | 14.0 s/veh | Met | 1 |
| Random search | 15.8 s/veh (14.9–17.0) | 15.0 s/veh (12.7–16.1) | 3/3 met | 8 |
| Grid optimum | 12.8 s/veh | 11.2 s/veh | Met | 150 |

### G3: Lowest delay on the worst approach.

| Method | Best result | Unused seeds | Goal | Runs |
| --- | --- | --- | --- | --- |
| Agent | not run yet | – | – | 0 of 3 repeats |
| Webster | 21.4 s/veh | 19.5 s/veh | Met | 1 |
| Random search | 31.6 s/veh (22.5–36.6) | 31.1 s/veh (20.9–34.0) | 3/3 met | 8 |
| Grid optimum | 17.6 s/veh | 17.9 s/veh | Met | 150 |

## Note on the free tier

A 429 from Groq's free tier ended 7 agent run(s) early (2 of them by the daily token quota; 5 were recorded before the kind of limit was saved, so for those it is not known). Those runs were set aside and run again from the start; they are not in the tables above. None of the runs in the tables needed a wait for a per-minute limit. A wait, when one happens, does not change what the model sees or how many calls it gets. The daily quota is not waited out: the stage stops and resumes on a later day.

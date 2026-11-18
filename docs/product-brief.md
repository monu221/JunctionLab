# JunctionLab – Product Brief

JunctionLab turns a plain-language description of a road junction into a running SUMO simulation and reports delay, queues and level of service, so a planner can compare two designs in minutes instead of hours.

## Problem

Setting up even a simple junction in a traffic simulator means writing network, demand and signal files by hand. That takes hours and specialist knowledge, so quick design questions ("what if we add a bus lane?") rarely get simulated.

## Users

- Transport engineering students and junior planners who need a quick first estimate
- Planners in early design stages comparing options before a full study

## What version 1 does

1. The user describes a junction in text, for example "four-arm junction, two lanes per approach, 800 veh/h from the north, fixed-time signal with a 90 s cycle".
2. An LLM turns the text into a structured scenario, shown as an editable form so the user can check and correct it.
3. The tool generates the SUMO files and runs the simulation 5 times with different random seeds.
4. It shows average delay, queue length and level of service per approach, plus a drawing of the junction.
5. The user can copy a scenario, change one thing, and compare the two side by side.
6. A one-page report downloads as PDF.

## Out of scope for version 1

- Pedestrians and cyclists
- Networks of several junctions
- Adaptive signal control
- Importing real junctions from map data
- Animation

## Success metric

- A described junction is simulated and reported without the user touching a SUMO file.
- For a simple undersaturated fixed-time junction, the simulated delay agrees with Webster's formula within a stated margin.
- The three bundled examples run without any API call.

## Stack

Python 3.12, eclipse-sumo and sumolib (pip), Claude API (small model), Pydantic, Streamlit, Plotly, fpdf2, pytest, GitHub Actions.

## Key design decisions

- The LLM only fills a structured scenario; it never writes SUMO files. Python generates the files with fixed, tested code.
- The user confirms the scenario before it runs.
- Every scenario runs with 5 seeds; results are reported as mean and range.
- Bundled examples work without an API key.
- A daily limit on LLM calls in the app, and a monthly spending limit on the API account.

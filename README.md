# JunctionLab
Plain-language junction descriptions to SUMO traffic simulations

See [docs/product-brief.md](docs/product-brief.md), [docs/method.md](docs/method.md) and [docs/plan.md](docs/plan.md).

## Development

Python 3.12. SUMO comes from the `eclipse-sumo` pip package, so no separate install is needed.

```sh
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

Defaults live in `config.yaml`.

## Command line

```sh
python -m junctionlab list                      # bundled examples
python -m junctionlab run four_arm_signal       # simulate 5 seeds, print delay/queue/throughput/LOS
python -m junctionlab run my_scenario.json --seeds 1,2 --out runs/test
```

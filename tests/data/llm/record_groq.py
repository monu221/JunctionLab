"""Re-record the Groq responses used by tests/test_llm.py. Makes live API calls.

Needs GROQ_API_KEY in .env. Run from the repo root:
    python tests/data/llm/record_groq.py [name ...]

Waits between calls to stay inside the free tier's tokens-per-minute limit. The key is only
sent as a request header; recordings hold the description and the response body.
"""

import contextlib
import json
import sys
import tempfile
import time
from pathlib import Path

from junctionlab import llm
from junctionlab.config import get_config

HERE = Path(__file__).parent
PAUSE_S = 15


def example(key: str) -> str:
    path = HERE.parents[2] / "examples" / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8"))["description"]


CASES = {
    "four_arm_signal": example("four_arm_signal"),
    "t_junction_priority": example("t_junction_priority"),
    "four_arm_bus_lane": example("four_arm_bus_lane"),
    "movement_counts": (
        "Signalised crossroads, 100 second cycle, one approach at a time. Peak 15-minute counts: "
        "from the north 25 left, 150 straight, 25 right; from the south 20 left, 120 straight, "
        "20 right; from the east 100 vehicles, a third of them turning right and the rest going "
        "straight; from the west 90 vehicles. Single lane on each arm, 50 km/h. There is a "
        "pedestrian crossing on the west arm."
    ),
    "partial_shares": (
        "T-junction with a stop-controlled side road from the north. Main road east-west, "
        "600 veh/h from the east and 550 veh/h from the west, side road 200 veh/h. "
        "On the main road 10% turn into the side road from the west."
    ),
    "unclear_demand": (
        "A busy signalised crossroads near the station, worst in the evening rush hour."
    ),
    "not_a_junction": "Write me a poem about the sea.",
    "invalid_cycle": (
        "Crossroads with traffic lights on a 15 second cycle, 300 veh/h on every approach."
    ),
    "max_tokens": example("four_arm_signal"),  # recorded with a tiny token limit
    "auth_error": "any",  # recorded with an invalid key
}


def record(name: str, text: str) -> int:
    cfg = get_config()
    key = None
    if name == "max_tokens":
        groq = cfg.llm.groq.model_copy(update={"max_tokens": 20})
        cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"groq": groq})})
    if name == "auth_error":
        key = "invalid-key-for-recording"
    path = HERE / f"{name}.json"
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "recording.json"
        transport = llm.RecordingTransport(target, text)
        client = llm.make_client(key, transport=transport, max_retries=0, provider="groq")
        with contextlib.suppress(llm.LlmError):  # several cases are recorded failures
            llm.call_llm(text, client, cfg)
        recording = json.loads(target.read_text(encoding="utf-8"))
    status = recording["response"]["status"]
    if status == 429:
        return int(float(recording["response"].get("headers", {}).get("retry-after", 30))) + 1
    path.write_text(json.dumps(recording, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: HTTP {status}")
    return 0


def main(names: list[str]) -> None:
    for i, name in enumerate(names or CASES):
        if i:
            time.sleep(PAUSE_S)
        for _ in range(4):
            wait = record(name, CASES[name])
            if not wait:
                break
            print(f"{name}: rate-limited, waiting {wait} s")
            time.sleep(wait)
        else:
            sys.exit(f"{name}: still rate-limited")


if __name__ == "__main__":
    main(sys.argv[1:])

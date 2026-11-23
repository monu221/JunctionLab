"""Re-record the agent session used by tests/test_agent.py. Makes live Groq calls.

The simulator is the deterministic fake in tests/agent_fake.py, so replay gets the same tool
results that the model saw. Needs GROQ_API_KEY in .env. Run from the repo root:
    python tests/data/agent/record_session.py

The key is only sent as a request header; the recording holds response bodies.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx2

sys.path.insert(0, str(Path(__file__).parents[2]))

from agent_fake import SESSION, fake_summary  # noqa: E402

from junctionlab import agent, llm  # noqa: E402
from junctionlab.examples import EXAMPLES_DIR, load_example  # noqa: E402

GOAL = "All approaches LOS C or better, shortest cycle possible."


class SessionRecorder(httpx2.HTTPTransport):
    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        response = super().handle_request(request)
        body = response.read()
        self.calls.append({"status": response.status_code, "body": json.loads(body)})
        headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
        }
        return httpx2.Response(response.status_code, headers=headers, content=body)


def main() -> None:
    transport = SessionRecorder()
    client = llm.make_client(transport=transport, max_retries=0, provider="groq")
    scenario = load_example(EXAMPLES_DIR / "four_arm_signal.json").scenario()
    goal = agent.read_goal(GOAL, client)
    result = agent.run_agent(scenario, goal, client, simulate=fake_summary)
    if result.stop_reason == "rate_limited":
        sys.exit("rate-limited: nothing saved, try again in a minute")
    recording = {
        "source": "live",
        "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "goal": GOAL,
        "calls": transport.calls,
    }
    SESSION.write_text(json.dumps(recording, indent=1) + "\n", encoding="utf-8")
    print(
        f"{len(transport.calls)} calls, {result.runs_used} runs, {result.stop_reason}, "
        f"{result.status}, best {agent.design_label(result.best)}"
    )


if __name__ == "__main__":
    main()

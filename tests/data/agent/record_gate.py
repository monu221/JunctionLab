"""M9 gate: one live Groq tool call with the agent's tool shapes. Makes one API call.

Needs GROQ_API_KEY in .env. Run from the repo root:
    python tests/data/agent/record_gate.py

The key is only sent as a request header; the recording holds the prompt and response body.
"""

import json
import sys
from pathlib import Path

from junctionlab import llm
from junctionlab.config import get_config

PATH = Path(__file__).parent / "tool_call_gate.json"
REASON = {"type": "string", "description": "One line: why this step."}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "propose_change",
            "description": "Create a new signal design from the current one. Give only the "
            "fields to change; use null for the others. Greens plus 5 s intergreen per phase "
            "must add up to the cycle.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": REASON,
                    "cycle_s": {"type": ["number", "null"]},
                    "phase_plan": {
                        "type": ["string", "null"],
                        "enum": ["two_phase", "four_phase_split", None],
                    },
                    "green_s": {"type": ["array", "null"], "items": {"type": "number"}},
                },
                "required": ["reason", "cycle_s", "phase_plan", "green_s"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate",
            "description": "Simulate a design by id. Uses one run from the budget.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"reason": REASON, "design_id": {"type": "string"}},
                "required": ["reason", "design_id"],
                "additionalProperties": False,
            },
        },
    },
]
PROMPT = (
    "Goal: every approach LOS C or better, with the shortest cycle possible.\n"
    "Current design D0: two_phase, cycle 90 s, greens [49, 31] s. Simulated: north 21 s, "
    "south 18 s, east 34 s, west 35 s control delay. Runs used: 1 of 8.\n"
    "Propose the next design."
)


def main() -> None:
    cfg = get_config()
    body = {
        "model": cfg.llm.groq.model,
        "max_completion_tokens": cfg.llm.groq.max_tokens,
        "temperature": 0,
        "reasoning_effort": cfg.llm.groq.reasoning_effort,
        "include_reasoning": False,
        "messages": [
            {"role": "system", "content": "You design fixed-time signals. Act through tools."},
            {"role": "user", "content": PROMPT},
        ],
        "tools": TOOLS,
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    transport = llm.RecordingTransport(PATH, PROMPT)
    client = llm.make_client(transport=transport, max_retries=0, provider="groq")
    response = client.post_chat(body)
    print("HTTP", response.status_code)
    data = response.json()
    if response.status_code != 200:
        sys.exit(f"gate failed: {json.dumps(data.get('error', data))[:500]}")
    choice = data["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    print("finish_reason", choice.get("finish_reason"), "| tool calls", len(calls))
    print("usage", data.get("usage"))
    if len(calls) != 1:
        sys.exit("gate failed: expected exactly one tool call")
    name = calls[0]["function"]["name"]
    args = json.loads(calls[0]["function"]["arguments"])
    print(name, json.dumps(args, ensure_ascii=True))
    schema = next(t for t in TOOLS if t["function"]["name"] == name)["function"]["parameters"]
    if set(args) != set(schema["required"]):
        sys.exit("gate failed: arguments do not match the schema")
    print("gate passed")


if __name__ == "__main__":
    main()

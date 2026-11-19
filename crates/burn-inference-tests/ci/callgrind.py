"""Counts the instructions of one request per scenario with Callgrind.

Each scenario runs twice, with zero and with `--requests` measured requests, and the difference is
divided by the count, which removes loading and warmup from the result.
"""

import argparse
import json
import os
import re
import subprocess

SCENARIOS = ["mobilenet_v2/b1", "resnet50/b1", "minilm/b4", "distilgpt2/prefill"]


def instructions(bench, scenario, iterations):
    result = subprocess.run(
        [
            "valgrind",
            "--tool=callgrind",
            "--callgrind-out-file=/dev/null",
            bench,
            "--scenario",
            scenario,
            "--warmup",
            "1",
            "--iterations",
            str(iterations),
        ],
        check=True,
        capture_output=True,
        text=True,
        # One worker keeps the count independent of how work is split between threads.
        env={**os.environ, "RAYON_NUM_THREADS": "1"},
    )
    return int(re.search(r"Collected : (\d+)", result.stderr)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--requests", type=int, default=2)
    args = parser.parse_args()

    entries = []
    print("| Scenario | Instructions per request |\n| --- | ---: |")
    for scenario in args.scenarios.split(","):
        baseline = instructions(args.bench, scenario, 0)
        measured = instructions(args.bench, scenario, args.requests)
        per_request = (measured - baseline) / args.requests
        entries.append({"name": f"flex {scenario} instructions", "unit": "instructions", "value": per_request})
        print(f"| {scenario} | {per_request:,.0f} |", flush=True)

    json.dump(entries, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()

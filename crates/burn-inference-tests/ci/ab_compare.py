"""Compares the latency of two lab-bench builds on one machine.

The builds run alternately, scenario by scenario, so that load from other tenants of a shared runner
affects both. A scenario regresses when the head's median p50 exceeds the base's by more than the
threshold, and by more than the absolute minimum.
"""

import argparse
import json
import os
import statistics
import subprocess
import tempfile

SCENARIOS = [
    "resnet50/b1",
    "resnet50/b8",
    "mobilenet_v2/b1",
    "mobilenet_v2/b16",
    "minilm/b4",
    "distilgpt2/prefill",
    "distilgpt2/decode16",
]


def p50(binary, scenario, iterations):
    with tempfile.NamedTemporaryFile(suffix=".json") as output:
        subprocess.run(
            [binary, "--scenario", scenario, "--iterations", str(iterations), "--json", output.name],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        entries = json.load(open(output.name))
    return next(entry["value"] for entry in entries if entry["name"].endswith(" p50"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--min-delta-ms", type=float, default=0.5)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    scenarios = args.scenarios.split(",")
    samples = {(build, scenario): [] for build in ("base", "head") for scenario in scenarios}
    for round_index in range(args.rounds):
        order = [("base", args.base), ("head", args.head)]
        if round_index % 2:
            order.reverse()
        for scenario in scenarios:
            for build, binary in order:
                samples[build, scenario].append(p50(binary, scenario, args.iterations))

    lines = [
        "| Scenario | Base p50 (ms) | Head p50 (ms) | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    regressed = []
    for scenario in scenarios:
        base = statistics.median(samples["base", scenario])
        head = statistics.median(samples["head", scenario])
        change = head / base - 1
        slower = change > args.threshold and head - base > args.min_delta_ms
        if slower:
            regressed.append(scenario)
        mark = " **regressed**" if slower else ""
        lines.append(f"| {scenario} | {base:.2f} | {head:.2f} | {change:+.1%}{mark} |")

    lines.append("")
    lines.append(
        f"Median of {args.rounds} alternating rounds of {args.iterations} requests. "
        f"Regression: more than {args.threshold:.0%} and {args.min_delta_ms} ms slower."
    )
    report = "\n".join(lines) + "\n"
    open(args.report, "w").write(report)
    print(report)

    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as file:
            file.write(f"regressed={'true' if regressed else 'false'}\n")


if __name__ == "__main__":
    main()

"""Lists the distinct lab-fuzz findings of a run as a Markdown table, for the job summary.

Findings are leads, not confirmed bugs: issues are opened by hand once a bug is confirmed.

PyTorch's verdict, from `verdicts.jsonl` when present, is listed with each finding. A program where
only Flex disagrees with PyTorch is a Flex bug whatever the backend under test, so its fingerprint
drops the backend and the same bug found on several backends is listed once.

Usage: summarize_findings.py <findings directory> <run URL>
"""

import json
import sys
from pathlib import Path


def verdicts():
    path = Path("verdicts.jsonl")
    if not path.exists():
        return {}
    lines = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {verdict["program"]: verdict["verdict"] for verdict in lines}


def cell(value):
    return str(value).replace("\n", " ").replace("|", "\\|")


def main(findings, run_url):
    judged = verdicts()
    rows = {}
    for path in sorted(Path(findings).glob("**/*.json")):
        if path.name.endswith(".program.json"):
            continue
        finding = json.loads(path.read_text())
        program = str(path.with_name(path.name.removesuffix(".json") + ".program.json"))
        verdict = judged.get(program, "unknown")
        fingerprint = finding["fingerprint"]
        if verdict == "flex-wrong":
            operation = fingerprint.split("-")[-1]
            fingerprint = f"flex-reference-{finding['failure']}-{operation}"
        rows.setdefault(
            fingerprint,
            [
                fingerprint,
                verdict,
                finding.get("dtype", "f32"),
                finding["seed"],
                finding["detail"][:200],
                f"`{finding['reproduce']}`",
            ],
        )

    print("## lab-fuzz findings\n")
    if not rows:
        print("No findings.")
        return
    print(
        f"{len(rows)} distinct findings, not yet confirmed. Programs and outputs are in the "
        f"`findings-*` artifacts of {run_url}.\n"
    )
    print("| Fingerprint | PyTorch verdict | Dtype | Seed | Failure | Reproduce |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows.values():
        print("| " + " | ".join(cell(value) for value in row) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:3])

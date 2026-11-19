# /// script
# requires-python = ">=3.11"
# dependencies = ["cbor2"]
# ///
"""Triages lab-fuzz findings on this machine, from a finding to an issue draft.

Local only: it builds Burn several times, drives the GPU and calls `gh`, so no workflow runs it.
For each distinct finding it:

1. replays the case on upstream `origin/main`, in fresh processes, and counts how often it fails;
2. replays it on `origin/main` with the unmerged `fix/*` branches merged, and when that passes, finds
   which branch fixes it: the finding is then a known bug;
3. shrinks the CubeCL environment to the autotune results the failure needs;
4. shrinks the case with `lab-fuzz reduce`, and asks `ci/oracle.py` which side PyTorch agrees with;
5. writes the reduced case as a standalone crate with `lab-fuzz emit`, builds it against the
   upstream checkout and runs it;
6. optionally bisects upstream commits with that crate;
7. searches the issues and pull requests of upstream and the fork for duplicates;
8. writes `report.md` with the evidence for each condition an issue needs, and `summary.md`.

It never opens an issue: whether the evidence is enough is for a person to decide.

Builds take tens of gigabytes, so once the reports are written it deletes the worktrees, their
builds and the intermediate files, keeping each report with its reduced case, reproduction and
the autotune results it needs. `--keep-trees` keeps the worktrees and builds for the next run;
`--clean` only deletes them.

Usage: uv run triage.py [--patched BRANCH]... [--no-patched] [--bisect] [--keep-trees] [--out DIR]
                 [--trees DIR] FINDINGS_DIR...
       uv run triage.py --clean [--trees DIR]
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
UPSTREAM = "tracel-ai/burn"
FORK = "kongxinjun/burn"
REPLAYS = 3
LAB_IGNORED = shutil.ignore_patterns("findings", "target", "__pycache__", "environment-*")


def log(message):
    print(message, file=sys.stderr, flush=True)


def run(command, cwd=None, env=None, timeout=None, check=True):
    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(map(str, command))} failed ({result.returncode}):\n{result.stderr[-3000:]}"
        )
    return result


def git(*args, cwd=REPO, check=True):
    return run(["git", *args], cwd=cwd, check=check).stdout.strip()


class Tree:
    """A worktree of upstream `origin/main`, with branches merged in and the lab crate copied in."""

    def __init__(self, root, name, branches=()):
        self.path = root / name
        # Every worktree builds into one directory, so dependencies are built once for all.
        self.target = root / "target"
        self.branches = list(branches)
        self.binaries = {}
        self.prepared = False

    def prepare(self):
        if self.prepared:
            return self
        if not (self.path / ".git").exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            git("worktree", "add", "--detach", self.path, "origin/main")
        git("reset", "-q", "--hard", "origin/main", cwd=self.path)
        git("clean", "-qfdx", "-e", "target", cwd=self.path)
        for branch in self.branches:
            merged = run(
                ["git", "-c", "user.name=lab", "-c", "user.email=lab@localhost", "merge",
                 "--no-edit", "-q", branch],
                cwd=self.path,
                check=False,
            )
            if merged.returncode != 0:
                git("merge", "--abort", cwd=self.path, check=False)
                raise RuntimeError(f"{branch} does not merge into origin/main")
        lab = self.path / "crates" / "burn-inference-tests"
        shutil.rmtree(lab, ignore_errors=True)
        shutil.copytree(LAB, lab, ignore=LAB_IGNORED)
        self.prepared = True
        return self

    def lab_fuzz(self, features):
        """lab-fuzz built here with `features`, copied so other feature sets do not replace it."""
        if features not in self.binaries:
            self.prepare()
            log(f"Building lab-fuzz ({features}) in {self.path.name}")
            run(
                ["cargo", "build", "--release", "-p", "burn-inference-tests", "--bin", "lab-fuzz",
                 "--features", features],
                cwd=self.path,
                env=dict(os.environ, CARGO_TARGET_DIR=str(self.target)),
            )
            name = f"{self.path.name}-lab-fuzz-{features.replace(',', '-')}"
            binary = self.target / "lab-bin" / name
            binary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.target / "release" / "lab-fuzz", binary)
            self.binaries[features] = binary
        return self.binaries[features]


def features_of(backend):
    """Cargo features of a backend name such as `metal+fusion`."""
    return backend.replace("+", ",")


def replay(binary, case, environment=None, program=None):
    """What the last run of `case` does in a new process."""
    command = [binary, "replay", case]
    if environment is not None:
        command += ["--environment", environment]
    if program is not None:
        command += ["--program", program]
    with tempfile.TemporaryDirectory() as cwd:
        result = run(command, cwd=cwd, timeout=1800, check=False)
    line = next(
        (line[len("REPLAY "):] for line in result.stdout.splitlines() if line.startswith("REPLAY ")),
        None,
    )
    if line is None:
        tail = " / ".join(result.stderr.strip().splitlines()[-5:])
        return {"verdict": "crashed", "panicked": True, "detail": f"{result.returncode}: {tail}"}
    return json.loads(line)


def message_hash(message):
    """lab-fuzz's hash of a panic message, ignoring the numbers in it."""
    first_lines = "".join(message.splitlines()[:12])
    value = 0x811C9DC5
    for byte in first_lines.encode():
        if chr(byte).isdigit():
            continue
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def fails_like(outcome, expected):
    if outcome["verdict"] != expected["verdict"]:
        return False
    if outcome["verdict"] == "crashed":
        return True
    if outcome["verdict"] != "failed" or outcome["panicked"] != expected["panicked"]:
        return False
    return not outcome["panicked"] or message_hash(outcome["detail"]) == message_hash(
        expected["detail"]
    )


def failure_rate(binary, case, expected, environment=None, times=REPLAYS):
    return sum(fails_like(replay(binary, case, environment), expected) for _ in range(times))


def minimize(items, fails):
    """Delta debugging: a small subsequence of `items` for which `fails` holds."""
    parts = 2
    while len(items) >= 2:
        chunk = -(-len(items) // parts)
        for start in range(0, len(items), chunk):
            complement = items[:start] + items[start + chunk:]
            if fails(complement):
                items, parts = complement, max(parts - 1, 2)
                break
        else:
            if parts >= len(items):
                break
            parts = min(parts * 2, len(items))
    if len(items) == 1 and fails([]):
        return []
    return items


def database(environment):
    databases = sorted(Path(environment).glob("*.db"))
    if len(databases) != 1:
        raise RuntimeError(f"{environment} should hold one CubeCL database, not {len(databases)}")
    return databases[0]


def autotune_entries(environment):
    with sqlite3.connect(database(environment)) as connection:
        return connection.execute(
            "SELECT namespace, key FROM entries WHERE namespace LIKE 'autotune/%'"
        ).fetchall()


def environment_with(source, kept, destination):
    """A copy of `source` keeping only the autotune entries in `kept`."""
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    for path in Path(source).iterdir():
        if path.is_file() and not path.name.endswith("-shm"):
            shutil.copy2(path, destination / path.name)
    kept = set(kept)
    with sqlite3.connect(database(destination)) as connection:
        removed = [entry for entry in autotune_entries(destination) if entry not in kept]
        connection.executemany("DELETE FROM entries WHERE namespace = ? AND key = ?", removed)
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return destination


def reduce_environment(binary, case, expected, environment, work):
    """The autotune entries of `environment` the failure needs, or None when it needs none."""
    if fails_like(replay(binary, case, "none"), expected):
        return None, []
    # Build records do not change what runs: drop them once, so each candidate copies less.
    environment = environment_with(environment, autotune_entries(environment), work / "base")
    with sqlite3.connect(database(environment)) as connection:
        connection.execute("DELETE FROM entries WHERE namespace LIKE 'records/%'")
        connection.commit()
        connection.execute("VACUUM")
    entries = autotune_entries(environment)
    candidate = work / "environment-candidate"

    def fails(subset):
        return fails_like(replay(binary, case, environment_with(environment, subset, candidate)), expected)

    # Whole namespaces first, then entries within the ones kept.
    namespaces = sorted({namespace for namespace, _ in entries})
    namespaces = minimize(
        namespaces, lambda kept: fails([e for e in entries if e[0] in set(kept)])
    )
    entries = [entry for entry in entries if entry[0] in set(namespaces)]
    entries = minimize(entries, fails)
    reduced = environment_with(environment, entries, work / "environment")
    return reduced, entries


def describe_entries(entries):
    """The namespace and key of each autotune entry, decoded when cbor2 is installed."""
    try:
        import cbor2
    except ImportError:
        cbor2 = None
    lines = []
    for namespace, key in entries:
        decoded = cbor2.loads(key).get("key", key) if cbor2 else f"<{len(key)} bytes of CBOR>"
        lines.append(f"- `{namespace.rsplit('/', 1)[-1]}`: `{decoded}`")
    return lines


def oracle(program):
    result = run(
        ["uv", "run", "-q", LAB / "ci" / "oracle.py", program], check=False, timeout=900
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1])["verdict"] if lines else "oracle-failed"


def standalone(binary, case, environment, burn, directory, target):
    """Emits `case` against the checkout `burn`, builds and runs it: (reproduced, output)."""
    shutil.rmtree(directory, ignore_errors=True)
    command = [binary, "emit", case, "--out", directory, "--burn", burn]
    command += ["--environment", environment if environment else "none"]
    run(command)
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    built = run(["cargo", "build", "--release", "-q"], cwd=directory, env=env, check=False)
    if built.returncode != 0:
        return None, built.stderr[-2000:]
    result = run([target / "release" / "lab-repro"], cwd=directory, check=False, timeout=1800)
    # CubeCL keeps what the run tuned in the crate's `target`, which the report does not need.
    shutil.rmtree(directory / "target", ignore_errors=True)
    differ = re.search(r"(\d+) of (\d+) elements differ", result.stdout)
    if differ:
        return int(differ.group(1)) > 0, result.stdout.strip()
    # No comparison: the device panicked before it.
    panic = re.search(r"panicked at .*?\n(.*)", result.stderr)
    return True, f"panicked: {panic.group(1) if panic else result.stderr[-500:]}"


def bisect(binary, case, environment, trees, directory):
    """The first upstream commit the standalone reproduction fails on, as a sentence."""
    tree = trees / "bisect"
    if not (tree / ".git").exists():
        git("worktree", "add", "--detach", tree, "origin/main")
    commits = git("rev-list", "--first-parent", "--max-count=513", "origin/main").split()
    target = trees / "target"
    results = {}

    def test(index):
        if index not in results:
            git("checkout", "-q", "--detach", commits[index], cwd=tree)
            reproduced, _ = standalone(binary, case, environment, tree, directory, target)
            results[index] = {True: "bad", False: "good", None: "skip"}[reproduced]
            log(f"  {commits[index][:9]}: {results[index]}")
        return results[index]

    good = next((i for i in (8, 32, 128, 512) if i < len(commits) and test(i) == "good"), None)
    if good is None:
        return "Not a recent regression: it fails, or does not build, 512 commits back."
    bad = 0
    while good - bad > 1:
        middle = (good + bad) // 2
        verdict = test(middle)
        # A commit that does not build is stepped over towards the bad end.
        probe = middle
        while verdict == "skip" and probe - 1 > bad:
            probe -= 1
            verdict = test(probe)
        if verdict == "bad":
            bad = probe
        elif verdict == "good":
            good = probe
        else:
            break
    commit = commits[bad]
    subject = git("log", "-1", "--format=%s", commit)
    return f"First fails at {commit[:9]} ({subject}); passes at {commits[good][:9]}."


def duplicates(terms):
    """Issues and pull requests of upstream and the fork that mention `terms`."""
    found = {}
    for repository in (UPSTREAM, FORK):
        for kind in ("issues", "prs"):
            result = run(
                ["gh", "search", kind, "--repo", repository, "--limit", "5",
                 "--json", "number,title,state,url", "--", terms],
                check=False,
            )
            if result.returncode == 0:
                for item in json.loads(result.stdout):
                    found[item["url"]] = item
    return list(found.values())


def operation_name(op):
    """A search term for a serialized operation, such as `neg` or `sum_dim`."""
    if not isinstance(op, dict):
        return op.lower()
    name, value = next(iter(op.items()))
    if name == "Unary":
        return value.lower()
    if name == "Reduce":
        return f"{value['op'].lower()}_dim"
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def checkbox(done):
    return "x" if done else " "


def triage(finding_path, args, trees, upstream, patched, single_fixes):
    finding = json.loads(finding_path.read_text())
    fingerprint = finding["fingerprint"]
    out = args.out / fingerprint
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    features = features_of(finding["backend"])
    binary = upstream.lab_fuzz(features)
    report = {"fingerprint": fingerprint, "finding": finding, "status": None, "notes": []}

    case = finding_path.with_name(finding_path.name.removesuffix(".json") + ".case.json")
    if not case.exists():
        # Findings from before sessions were recorded: regenerate the case from the seed.
        log(f"{fingerprint}: no case, running seed {finding['seed']} again")
        options = re.findall(r"--(max-ops|dtypes) (\S+)", finding.get("reproduce", ""))
        command = [binary, "--seed", finding["seed"], "--programs", "1", "--findings", out / "rerun"]
        for name, value in options:
            command += [f"--{name}", value]
        with tempfile.TemporaryDirectory() as cwd:
            run(command, cwd=cwd, check=False, timeout=3600)
        cases = sorted((out / "rerun").glob("*.case.json"))
        if not cases:
            report["status"] = "not-reproduced"
            report["notes"].append("The seed passes here: no case to replay.")
            return write_report(out, report)
        case = cases[0]

    log(f"{fingerprint}: replaying on origin/main {upstream.base[:9]}")
    expected = replay(binary, case)
    if expected["verdict"] not in ("failed", "crashed"):
        report["status"] = "not-reproduced"
        report["notes"].append(f"The case passes on origin/main: {expected['detail'] or 'no failure'}.")
        return write_report(out, report)
    report["detail"] = expected["detail"]
    report["rate"] = 1 + failure_rate(binary, case, expected, times=REPLAYS - 1)

    if patched is not None:
        if failure_rate(patched.lab_fuzz(features), case, expected, times=REPLAYS) == 0:
            fixed_by = [
                branch
                for branch, tree in single_fixes.items()
                if failure_rate(tree.lab_fuzz(features), case, expected, times=REPLAYS) == 0
            ]
            report["status"] = "known"
            report["fixed_by"] = fixed_by or patched.branches
            return write_report(out, report)

    log(f"{fingerprint}: shrinking the environment")
    work = out / "work"
    work.mkdir()
    case_data = json.loads(case.read_text())
    environment = None
    if case_data.get("environment"):
        source = (case.parent / case_data["environment"]).resolve()
        environment, entries = reduce_environment(binary, case, expected, source, work)
        report["environment"] = describe_entries(entries)
        if environment is not None:
            shutil.move(environment, out / "environment")
            environment = out / "environment"

    log(f"{fingerprint}: reducing the case")
    reduced = out / "reduced.case.json"
    run(
        [binary, "reduce", case, "--out", reduced,
         "--environment", environment if environment else "none"],
        timeout=7200,
    )
    reduced_data = json.loads(reduced.read_text())
    target = reduced_data["programs"][reduced_data["runs"][-1]["program"]]["program"]
    report["program"] = target
    report["runs"] = len(reduced_data["runs"])
    report["reduced_rate"] = failure_rate(binary, reduced, expected, environment or "none", times=5)
    program = out / "reduced.program.json"
    replay(binary, reduced, environment or "none", program)
    report["verdict"] = oracle(program) if program.exists() else "no-program"

    log(f"{fingerprint}: building the standalone reproduction")
    reproduced, output = standalone(
        binary, reduced, environment, upstream.path, out / "repro", trees / "target"
    )
    report["standalone"] = reproduced
    report["standalone_output"] = output

    if args.bisect and reproduced:
        log(f"{fingerprint}: bisecting")
        report["bisect"] = bisect(binary, reduced, environment, trees, out / "repro-bisect")

    last = operation_name(target["ops"][-1])
    backend = finding["backend"].split("+")[0]
    report["duplicates"] = duplicates(f"{last} {backend}") + duplicates(last)

    verdict = report["verdict"]
    if not reproduced:
        report["status"] = "lab-only"
    elif verdict == "flex-wrong":
        report["status"] = "flex-bug"
    elif verdict == "device-wrong":
        report["status"] = "candidate"
    elif verdict == "both-wrong":
        # PyTorch disagrees with Flex too: often an edge case such as infinities, to read by hand.
        report["status"] = "review"
    else:
        report["status"] = "noise"
    return write_report(out, report)


def discard_intermediates(out):
    """Deletes what a finding's report does not need: reduction work and regenerated runs."""
    for name in ("work", "rerun", "repro-bisect"):
        shutil.rmtree(out / name, ignore_errors=True)


def clean(trees):
    """Deletes the worktrees, every build in them and the environments replays left behind."""
    # Only what triage creates, in case `--trees` names a directory holding something else.
    created = ("upstream", "patched", "bisect", "target")
    paths = sorted(trees.iterdir()) if trees.exists() else []
    for path in paths:
        if path.name not in created and not path.name.startswith("fix-"):
            continue
        if (path / ".git").exists():
            git("worktree", "remove", "--force", path, check=False)
        shutil.rmtree(path, ignore_errors=True)
    if trees.exists() and not any(trees.iterdir()):
        trees.rmdir()
    git("worktree", "prune")
    # A replay that crashed leaves its scratch environment in the temporary directory.
    for path in Path(tempfile.gettempdir()).glob("lab-environment-*"):
        shutil.rmtree(path, ignore_errors=True)
    for path in Path(tempfile.gettempdir()).glob("lab-fuzz-*.case.json"):
        path.unlink(missing_ok=True)
    log(f"Deleted {trees} and the scratch environments")


def write_report(out, report):
    finding = report["finding"]
    lines = [f"# {report['fingerprint']}", "", f"Status: **{report['status']}**", ""]
    lines += [f"- Backend: {finding['backend']}, dtype {finding.get('dtype', 'f32')}"]
    lines += [f"- Fuzzer: seed {finding['seed']}, `{finding.get('detail', '')[:200]}`"]
    lines += [f"- {note}" for note in report["notes"]]
    if report["status"] == "known":
        lines += [f"- Fixed by: {', '.join(report['fixed_by'])}"]
    if "rate" in report:
        lines += ["", "## Conditions for an upstream issue", ""]
        standalone_ok = bool(report.get("standalone"))
        lines += [
            f"- [{checkbox(standalone_ok)}] Standalone reproduction on origin/main: "
            f"`repro/` ({report.get('standalone_output', 'not built')[:300]!r})",
            f"- [{checkbox(report.get('verdict') == 'device-wrong')}] Independent reference: "
            f"PyTorch verdict `{report.get('verdict')}`",
            f"- [ ] Not the environment or tolerance (check by hand): fails {report['rate']}/{REPLAYS} "
            f"replays, the reduced case {report.get('reduced_rate', '?')}/5",
            "- [ ] No duplicate (check the candidates below by hand)",
        ]
    if "program" in report:
        program = report["program"]
        ops = " -> ".join(json.dumps(op, separators=(",", ":"))[:120] for op in program["ops"])
        lines += ["", "## Reduced case", ""]
        lines += [f"- {report['runs']} run(s), dtype {program['dtype']}, input {program['input']['shape']}"]
        if program.get("resized"):
            lines += [f"- Other batch: {program['resized']['shape']}"]
        lines += [f"- Operations: {ops}"]
    if report.get("environment") is not None:
        lines += ["", "## Autotune results the failure needs", ""]
        lines += report["environment"] or ["None: it fails from an empty environment."]
    if "bisect" in report:
        lines += ["", "## Bisection", "", report["bisect"]]
    if "duplicates" in report:
        lines += ["", "## Possible duplicates", ""]
        lines += [
            f"- [{item['state'].lower()}] {item['title']} ({item['url']})"
            for item in report["duplicates"]
        ] or ["None found."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    discard_intermediates(out)
    log(f"{report['fingerprint']}: {report['status']}")
    return report


def findings(directories):
    """One finding per fingerprint, preferring those with a recorded case."""
    chosen = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("**/*.json")):
            if path.name.endswith((".program.json", ".case.json")):
                continue
            fingerprint = json.loads(path.read_text()).get("fingerprint")
            if fingerprint is None:
                continue
            has_case = path.with_name(path.name.removesuffix(".json") + ".case.json").exists()
            if fingerprint not in chosen or (has_case and not chosen[fingerprint][1]):
                chosen[fingerprint] = (path, has_case)
    return [path for path, _ in chosen.values()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("findings", nargs="*", type=Path)
    parser.add_argument("--patched", action="append", help="Unmerged fix branches to rule out")
    parser.add_argument("--no-patched", action="store_true")
    parser.add_argument("--bisect", action="store_true", help="Bisect upstream commits")
    parser.add_argument("--keep-trees", action="store_true", help="Keep worktrees and builds")
    parser.add_argument("--clean", action="store_true", help="Only delete worktrees and builds")
    parser.add_argument("--out", type=Path, default=REPO / "target" / "lab-triage")
    parser.add_argument("--trees", type=Path, default=REPO.parent / "burn-lab-trees")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.clean:
        clean(args.trees)
        return
    if not args.findings:
        parser.error("give the findings directories to triage")
    try:
        run_triage(args)
    finally:
        if not args.keep_trees:
            clean(args.trees)


def run_triage(args):
    git("fetch", "-q", "origin")
    upstream = Tree(args.trees, "upstream").prepare()
    upstream.base = git("rev-parse", "HEAD", cwd=upstream.path)
    branches = args.patched or git(
        "branch", "--list", "fix/*", "--no-merged", "origin/main", "--format=%(refname:short)"
    ).split()
    patched, single_fixes = None, {}
    if branches and not args.no_patched:
        patched = Tree(args.trees, "patched", branches).prepare()
        single_fixes = {
            branch: Tree(args.trees, "fix-" + branch.replace("/", "-"), [branch])
            for branch in branches
        }

    reports = []
    for path in findings(args.findings):
        try:
            reports.append(triage(path, args, args.trees, upstream, patched, single_fixes))
        except Exception as error:  # One broken finding should not stop the others.
            log(f"{path}: {error}")
            reports.append({"fingerprint": path.stem, "status": f"error: {str(error)[:200]}"})

    lines = ["# lab triage", "", f"origin/main {upstream.base[:9]}", ""]
    lines += ["| Fingerprint | Status | PyTorch | Standalone | Detail |", "| --- | --- | --- | --- | --- |"]
    order = ["candidate", "flex-bug", "review", "lab-only", "known", "noise", "not-reproduced"]
    for report in sorted(reports, key=lambda r: order.index(r["status"]) if r["status"] in order else 99):
        lines.append(
            f"| [{report['fingerprint']}]({report['fingerprint']}/report.md) | {report['status']} "
            f"| {report.get('verdict', '')} | {report.get('standalone', '')} "
            f"| {str(report.get('detail', '')).replace('|', '/')[:100]} |"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

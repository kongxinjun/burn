#!/usr/bin/env bash
# Files one GitHub issue per new lab-fuzz fingerprint. A finding whose issue is open gets a comment
# with the new seed; one whose issue was closed as completed reopens it; one closed as not planned
# is ignored. At most MAX_NEW_ISSUES issues are opened per run; the rest wait for the next one.
#
# PyTorch's verdict, from `verdicts.jsonl` when present, labels each issue. A program where only Flex
# disagrees with PyTorch is a Flex bug whatever the backend under test, so its fingerprint drops
# the backend and the same bug found on several backends gets one issue.
#
# Usage: report-findings.sh <findings directory>   (needs GH_TOKEN, GH_REPO and RUN_URL)
set -euo pipefail
shopt -s nullglob globstar

findings=${1:?findings directory}
max_new=${MAX_NEW_ISSUES:-5}
gh label create lab-fuzz --color B60205 --description "Found by lab-fuzz" --force > /dev/null

verdict_of() {
  local program=${1%.json}.program.json
  [[ -f verdicts.jsonl ]] || { echo unknown; return; }
  jq -r --arg program "$program" 'select(.program == $program) | .verdict' verdicts.jsonl | head -1 | grep . || echo unknown
}

declare -A reported
opened=0
for file in "$findings"/**/*.json; do
  [[ $file == *.program.json ]] && continue
  verdict=$(verdict_of "$file")
  fingerprint=$(jq -r .fingerprint "$file")
  if [[ $verdict == flex-wrong ]]; then
    fingerprint="flex-reference-$(jq -r '"\(.failure)-" + (.fingerprint | split("-") | last)' "$file")"
  fi
  [[ -n ${reported[$fingerprint]:-} ]] && continue
  reported[$fingerprint]=1

  body=$(jq -r --arg run "$RUN_URL" --arg verdict "$verdict" '
    "**Backend:** `\(.backend)`, \(.mode | ascii_downcase) execution in \(.dtype // "f32")\n" +
    "**Failure:** \(.failure): \(.detail | .[0:1500])\n" +
    "**PyTorch verdict:** \($verdict)\n" +
    "**Upstream Burn:** tracel-ai/burn@\(.burn_sha)\n" +
    "**Seed:** \(.seed), input shape \(.input_shape)\n" +
    "**Run:** \($run)\n\n" +
    "Shortest failing program (the last operation completes the failure):\n\n" +
    "```text\n\(.program | join("\n"))\n```\n\n" +
    "Reproduce in `crates/burn-inference-tests` on the lab branch, with upstream Burn at the commit above:\n\n" +
    "```sh\n\(.reproduce)\n```\n"' "$file")

  issue=$(gh issue list --label lab-fuzz --state all --limit 500 --json number,title,state,stateReason \
    --jq "map(select(.title == \"lab-fuzz: $fingerprint\")) | .[0] // empty")
  if [[ -z $issue ]]; then
    if (( opened >= max_new )); then
      echo "Deferred $fingerprint: $max_new issues already opened in this run"
      continue
    fi
    gh label create "verdict: $verdict" --color C5DEF5 --force > /dev/null
    gh issue create --label lab-fuzz --label "verdict: $verdict" --title "lab-fuzz: $fingerprint" --body "$body"
    opened=$((opened + 1))
    continue
  fi

  number=$(jq -r .number <<< "$issue")
  case "$(jq -r '.state + "/" + (.stateReason // "")' <<< "$issue")" in
    OPEN/*) gh issue comment "$number" --body "Seen again. $body" ;;
    CLOSED/NOT_PLANNED) echo "Ignoring $fingerprint: #$number was closed as not planned" ;;
    *) gh issue reopen "$number" --comment "Reappeared after being closed. $body" ;;
  esac
done
echo "${#reported[@]} distinct findings, $opened new issues"

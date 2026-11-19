#!/usr/bin/env bash
# Files one GitHub issue per new lab-fuzz fingerprint. A finding whose issue is open gets a comment
# with the new seed; one whose issue was closed as completed reopens it; one closed as not planned
# is ignored.
#
# Usage: report-findings.sh <findings directory>   (needs GH_TOKEN, GH_REPO and RUN_URL)
set -euo pipefail
shopt -s nullglob globstar

findings=${1:?findings directory}
gh label create lab-fuzz --color B60205 --description "Found by lab-fuzz" --force > /dev/null

declare -A reported
for file in "$findings"/**/*.json; do
  fingerprint=$(jq -r .fingerprint "$file")
  [[ -n ${reported[$fingerprint]:-} ]] && continue
  reported[$fingerprint]=1

  body=$(jq -r --arg run "$RUN_URL" '
    "**Backend:** `\(.backend)`, \(.mode | ascii_downcase) execution\n" +
    "**Failure:** \(.failure): \(.detail | .[0:1500])\n" +
    "**Upstream Burn:** tracel-ai/burn@\(.burn_sha)\n" +
    "**Seed:** \(.seed), input shape \(.input_shape)\n" +
    "**Run:** \($run)\n\n" +
    "Shortest failing program (the last operation completes the failure):\n\n" +
    "```text\n\(.program | join("\n"))\n```\n\n" +
    "Reproduce in `crates/burn-inference-tests` on the lab branch, with upstream Burn at the commit above:\n\n" +
    "```sh\n\(.reproduce)\n```\n"' "$file")

  issue=$(gh issue list --label lab-fuzz --state all --limit 200 --json number,title,state,stateReason \
    --jq "map(select(.title == \"lab-fuzz: $fingerprint\")) | .[0] // empty")
  if [[ -z $issue ]]; then
    gh issue create --label lab-fuzz --title "lab-fuzz: $fingerprint" --body "$body"
    continue
  fi

  number=$(jq -r .number <<< "$issue")
  case "$(jq -r '.state + "/" + (.stateReason // "")' <<< "$issue")" in
    OPEN/*) gh issue comment "$number" --body "Seen again. $body" ;;
    CLOSED/NOT_PLANNED) echo "Ignoring $fingerprint: #$number was closed as not planned" ;;
    *) gh issue reopen "$number" --comment "Reappeared after being closed. $body" ;;
  esac
done
echo "${#reported[@]} distinct findings reported"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINAL_SUBMISSION="${FINAL_SUBMISSION:-0}"
FAIL=0

required=(
  README.md
  docs/ENVIRONMENT_MATRIX.md
  docs/TECHNICAL_REPORT.md
  docs/REPRODUCIBILITY.md
  evidence/README.md
  output/pdf/datawhale-eai-radeon-physical-ai-technical-report.pdf
  SHA256SUMS
  THIRD_PARTY_NOTICES.md
  LICENSE
  submission/Track3-Datawhale-EAI/README.md
  submission/Track3-Datawhale-EAI/PR_BODY.md
  submission/Track3-Datawhale-EAI/PROJECT_PROFILE.md
)

for path in "${required[@]}"; do
  if [[ ! -s "$ROOT/$path" ]]; then
    echo "ERROR: missing or empty required file: $path" >&2
    FAIL=1
  fi
done

if [[ "$FINAL_SUBMISSION" == "1" ]]; then
  if grep -RInE '\b(TBD|TODO|PLACEHOLDER)\b' \
      "$ROOT/README.md" "$ROOT/docs/TECHNICAL_REPORT.md" \
      "$ROOT/docs/REPRODUCIBILITY.md" "$ROOT/docs/ENVIRONMENT_MATRIX.md" \
      "$ROOT/submission"; then
    echo "ERROR: unresolved placeholders remain in final-submission mode" >&2
    FAIL=1
  fi
fi

# Scan tracked and candidate text files. The expressions intentionally avoid
# printing matched secret values.
while IFS= read -r file; do
  case "$file" in
    .git/*|*.png|*.jpg|*.jpeg|*.gif|*.mp4|*.pdf) continue ;;
  esac
  if grep -Iq . "$ROOT/$file" 2>/dev/null && \
     grep -qE '(BEGIN (OPENSSH|RSA|EC) PRIVATE KEY|hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})' "$ROOT/$file"; then
    echo "ERROR: possible secret in $file" >&2
    FAIL=1
  fi
done < <(cd "$ROOT" && find . -type f -printf '%P\n')

large_files="$(cd "$ROOT" && find . -path './.git' -prune -o -type f -size +25M -printf '%P %s bytes\n')"
if [[ -n "$large_files" ]]; then
  echo "ERROR: files larger than 25 MiB must be externally hosted:" >&2
  echo "$large_files" >&2
  FAIL=1
fi

if [[ "$FAIL" -ne 0 ]]; then
  exit 1
fi

echo "Submission preflight passed (FINAL_SUBMISSION=$FINAL_SUBMISSION)."

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(2|4|8)$ ]]; then
    printf 'usage: %s {2|4|8}\n' "$0" >&2
    exit 2
fi

root_dir="$(git -C "$(dirname "$0")/../../.." rev-parse --show-toplevel)"
artifact_dir="$root_dir/.step32c/step32j-generation/gate-$1"
checkpoint="$artifact_dir/generation.checkpoints"
manifest="$artifact_dir/generation.manifest"
log="$artifact_dir/reference.current.log"
state="$artifact_dir/reference.state.json"
if [[ ! -s "$checkpoint" || ! -s "$manifest" ]]; then
    printf 'missing persistent production artifacts in %s\n' "$artifact_dir" >&2
    exit 1
fi

exec >> "$log" 2>&1
set +e
python -u "$root_dir/research/results/vbuf-ml-integration/step32g_bounded_reference.py" \
    "$manifest" \
    "$root_dir/.step32c/glm-4.5-air-fp8.vbuf" \
    "$checkpoint" \
    46 generation Test "$1" --state "$state"
status=$?
printf 'REFERENCE_EXIT_STATUS=%s\n' "$status"
date -Is > "$artifact_dir/reference.current-completed-at.tmp"
printf 'EXIT_STATUS=%s\n' "$status" >> "$artifact_dir/reference.current-completed-at.tmp"
mv "$artifact_dir/reference.current-completed-at.tmp" "$artifact_dir/reference.current-completed-at"
exit "$status"

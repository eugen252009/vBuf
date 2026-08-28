#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(2|4|8)$ ]]; then
    printf 'usage: %s {2|4|8}\n' "$0" >&2
    exit 2
fi

root_dir="$(git -C "$(dirname "$0")/../../.." rev-parse --show-toplevel)"
artifact_dir="$root_dir/.step32c/step32j-generation/gate-$1"
mkdir -p "$artifact_dir"
resume_arg=()
if [[ -s "$artifact_dir/generation.checkpoints.progress" ]] \
    && { \
        { grep -q '^PHASE=PREFILL$' "$artifact_dir/generation.checkpoints.progress" \
            && grep -q '^LAYER=45$' "$artifact_dir/generation.checkpoints.progress"; } \
        || grep -q '^PHASE=GENERATION$' "$artifact_dir/generation.checkpoints.progress" \
        || grep -q '^PHASE=GENERATION_COMPLETE$' "$artifact_dir/generation.checkpoints.progress"; \
    }; then
    resume_arg=(--resume-prefill)
fi

exec >> "$artifact_dir/run.log" 2>&1
set +e
"$root_dir/rust/target/debug/vbuf-runtime-step32j-generation" \
    "$root_dir/.step32c/glm-4.5-air-fp8.semantic.vbuf" \
    "$root_dir/.step32c/glm-4.5-air-fp8.vbuf" \
    "$artifact_dir/generation.checkpoints" \
    "$artifact_dir/generation.manifest" \
    Test "${resume_arg[@]}" "$1"
status=$?
printf 'HARNESS_EXIT_STATUS=%s\n' "$status"
date -Is > "$artifact_dir/completed-at.tmp"
printf 'EXIT_STATUS=%s\n' "$status" >> "$artifact_dir/completed-at.tmp"
mv "$artifact_dir/completed-at.tmp" "$artifact_dir/completed-at"
exit "$status"

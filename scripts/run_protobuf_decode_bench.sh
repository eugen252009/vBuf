#!/usr/bin/env sh
# Run one official isolated Prost benchmark report without overwriting evidence.
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 RUN_LABEL" >&2
    exit 64
fi

label=$1
case "$label" in
    *[!A-Za-z0-9_-]*|'') echo "RUN_LABEL must contain only A-Z, a-z, 0-9, _ or -" >&2; exit 64 ;;
esac

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cpu=${BENCH_CPU:-0}
report_dir="$root/benchmark-results/final"
report_path="$report_dir/protobuf-isolated-$label.csv"

if [ -e "$report_path" ]; then
    echo "refusing to overwrite existing report: $report_path" >&2
    exit 73
fi
mkdir -p "$report_dir"

affinity=$(taskset -c "$cpu" sh -c 'taskset -pc $$')
case "$affinity" in
    *": $cpu") ;;
    *) echo "failed to verify exact CPU affinity $cpu: $affinity" >&2; exit 1 ;;
esac

worktree_status=$(git -C "$root" status --short)
if [ -n "$worktree_status" ]; then
    echo "warning: worktree is dirty; raw report will record this" >&2
    worktree_status=$(printf '%s' "$worktree_status" | tr '\n' ';')
else
    worktree_status=clean
fi

effective_rustc_line=$(cd "$root/rust" && cargo rustc -vv --locked --release --bin protobuf_decode_bench -- --emit=metadata 2>&1 | grep 'rustc.*protobuf_decode_bench' || true)
if [ -z "$effective_rustc_line" ]; then
    echo "could not capture the effective rustc invocation" >&2
    exit 1
fi
effective_rustc_flags=$(printf '%s' "$effective_rustc_line" | sed 's/.*rustc --crate-name protobuf_decode_bench //')

source_sha256=$(sha256sum "$root/rust/src/bin/protobuf_decode_bench.rs" | cut -d ' ' -f 1)
rustc_version=$(rustc -Vv | tr '\n' ';')
rustflags=${RUSTFLAGS:-${CARGO_ENCODED_RUSTFLAGS:-<unset>}}

cd "$root/rust"
exec env \
    BENCH_RUN_LABEL="$label" \
    BENCH_COMMIT="$(git -C "$root" rev-parse HEAD)" \
    BENCH_SOURCE_SHA256="$source_sha256" \
    BENCH_RUSTC="$rustc_version" \
    BENCH_AFFINITY="$affinity" \
    BENCH_WORKTREE_STATUS="$worktree_status" \
    BENCH_RUSTFLAGS="$rustflags" \
    BENCH_EFFECTIVE_RUSTC_FLAGS="$effective_rustc_flags" \
    BENCH_REPORT_PATH="$report_path" \
    taskset -c "$cpu" cargo run --locked --release --bin protobuf_decode_bench

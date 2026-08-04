#!/usr/bin/env sh
# Run one non-overwriting real-vBuf SoA/AoS baseline report.
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 RUN_LABEL" >&2
    exit 64
fi
label=$1
case "$label" in *[!A-Za-z0-9_-]*|'') echo "invalid RUN_LABEL" >&2; exit 64 ;; esac

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cpu=${BENCH_CPU:-0}
report_dir="$root/benchmark-results/vbuf/final"
report_path="$report_dir/vbuf-baseline-$label.csv"
[ ! -e "$report_path" ] || { echo "refusing to overwrite $report_path" >&2; exit 73; }
mkdir -p "$report_dir"

affinity=$(taskset -c "$cpu" sh -c 'taskset -pc $$')
case "$affinity" in *": $cpu") ;; *) echo "failed to verify CPU $cpu: $affinity" >&2; exit 1 ;; esac
worktree_status=$(git -C "$root" status --short)
if [ -n "$worktree_status" ]; then echo "warning: dirty worktree recorded in raw report" >&2; worktree_status=$(printf '%s' "$worktree_status" | tr '\n' ';'); else worktree_status=clean; fi

preflight=$(mktemp -d "${TMPDIR:-/tmp}/vbuf-baseline-preflight.XXXXXX")
dataset_dir=$(mktemp -d "${TMPDIR:-/tmp}/vbuf-baseline-data.XXXXXX")
trap 'rm -rf "$preflight" "$dataset_dir"' EXIT HUP INT TERM
effective=$(cd "$root/rust" && cargo rustc -vv --locked --release --target-dir "$preflight" --bin vbuf_baseline_bench -- --emit=metadata 2>&1 | grep 'rustc.*vbuf_baseline_bench' || true)
[ -n "$effective" ] || { echo "could not capture effective rustc invocation" >&2; exit 1; }
effective=$(printf '%s' "$effective" | sed 's/.*rustc --crate-name vbuf_baseline_bench //')

runner_sha=$(sha256sum "$root/rust/src/bin/vbuf_baseline_bench.rs" | cut -d ' ' -f1)
core_sha=$(sha256sum "$root/rust/src/lib.rs" | cut -d ' ' -f1)
rustc_version=$(rustc -Vv | tr '\n' ';')
rustflags=${RUSTFLAGS:-${CARGO_ENCODED_RUSTFLAGS:-<unset>}}

cd "$root/rust"
env BENCH_RUN_LABEL="$label" BENCH_COMMIT="$(git -C "$root" rev-parse HEAD)" \
    BENCH_SOURCE_SHA256="$runner_sha" BENCH_VBUF_CORE_SHA256="$core_sha" BENCH_RUSTC="$rustc_version" \
    BENCH_AFFINITY="$affinity" BENCH_WORKTREE_STATUS="$worktree_status" BENCH_RUSTFLAGS="$rustflags" \
    BENCH_EFFECTIVE_RUSTC_FLAGS="$effective" BENCH_VBUF_SOA_PATH="$dataset_dir/soa.vbuf" \
    BENCH_VBUF_AOS_PATH="$dataset_dir/aos.vbuf" BENCH_REPORT_PATH="$report_path" \
    taskset -c "$cpu" cargo run --locked --release --bin vbuf_baseline_bench

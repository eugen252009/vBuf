#!/bin/sh
set -eu
label=${1:-step5a-raw}
out_dir=${BENCHMARK_RESULTS_DIR:-benchmark-results/vbuf-navigation}
raw="$out_dir/$label.csv"
meta="$out_dir/$label.environment.txt"
if [ -e "$raw" ] || [ -e "$meta" ]; then
  echo "refusing to overwrite existing report: $raw" >&2
  exit 1
fi
mkdir -p "$out_dir"
{
  printf 'commit='; git rev-parse HEAD
  printf 'rustc='; rustc -Vv | tr '\n' ';'; printf '\n'
  printf 'uname='; uname -a; printf '\n'
  printf 'cpu_model='; lscpu | awk -F: '/Model name/ {gsub(/^ +/, "", $2); print $2; exit}'
  printf 'architecture='; uname -m; printf '\n'
  printf 'rayon_threads=%s\n' "${RAYON_NUM_THREADS:-default}"
  printf 'runner_flags=release profile; see Cargo.toml and Cargo.lock\n'
  printf 'cache_note=warm in-process samples; cold mmap/page-fault qualification not claimed\n'
} > "$meta"
cargo run --release --manifest-path rust/Cargo.toml --bin v06_navigation_bench -- "$raw"
python3 scripts/validate_v06_navigation.py "$raw"

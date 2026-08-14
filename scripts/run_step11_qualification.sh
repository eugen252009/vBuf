#!/bin/sh
set -eu
out_dir=${BENCHMARK_RESULTS_DIR:-benchmark-results/vbuf-ml-step11}
label=${1:-step11-qualification}
raw="$out_dir/$label.csv"
env="$out_dir/$label.environment.txt"
fixture="$out_dir/$label.vbuf"
if [ -e "$raw" ] || [ -e "$env" ] || [ -e "$fixture" ]; then
  echo "refusing to overwrite existing report: $raw" >&2
  exit 1
fi
mkdir -p "$out_dir"
{
  printf 'commit='; git rev-parse HEAD
  printf 'runner_source_sha256='; sha256sum rust/vbuf-ml/examples/step11_qualification.rs scripts/run_step11_qualification.sh | sha256sum | awk '{print $1}'
  printf 'rustc='; rustc -Vv | tr '\n' ';'; printf '\n'
  printf 'uname='; uname -a; printf '\n'
  printf 'cache_note=real read-only mmap; warm page-cache/process runs; no drop_caches permission assumed; page-fault deltas are process counters, not storage-I/O proof\n'
  printf 'lazy=canonical validation + bootstrap + model metadata + tensor directory\n'
  printf 'eager=lazy + tokenizer control/content validation + all tokenizer arrays + auxiliary cold range\n'
} > "$env"
cargo run --release --manifest-path rust/Cargo.toml -p vbuf-ml --example step11_qualification -- "$fixture" > "$raw"
sha256sum "$fixture" >> "$env"
printf 'report=%s\n' "$raw"

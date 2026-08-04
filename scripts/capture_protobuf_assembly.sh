#!/usr/bin/env sh
# Capture the release assembly that corresponds to the checked-in runner hash.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source="$root/rust/src/bin/protobuf_decode_bench.rs"
sha=$(sha256sum "$source" | cut -d ' ' -f 1)
output_dir="$root/benchmark-results/assembly"
output="$output_dir/protobuf_decode_bench-$sha.s"

if [ -e "$output" ]; then
    echo "refusing to overwrite existing assembly artifact: $output" >&2
    exit 73
fi

mkdir -p "$output_dir"
(cd "$root/rust" && cargo rustc --locked --release --bin protobuf_decode_bench -- --emit=asm)
assembly=$(ls "$root"/rust/target/release/deps/protobuf_decode_bench-*.s | sort | tail -n 1)
cp "$assembly" "$output"
printf 'runner_source_sha256=%s\n' "$sha" > "$output.metadata"
printf 'runner_commit=%s\n' "$(git -C "$root" rev-parse HEAD)" >> "$output.metadata"
printf 'rustc=%s\n' "$(rustc -Vv | tr '\n' ';')" >> "$output.metadata"

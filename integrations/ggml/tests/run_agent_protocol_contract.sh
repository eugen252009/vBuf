#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
out="${TMPDIR:-/tmp}/vbuf-agent-protocol-contract-$$"
trap 'rm -f "$out"' EXIT
g++ -std=c++17 -Wall -Wextra -Werror \
  -I "$root/integrations/ggml/include" \
  "$root/integrations/ggml/src/vbuf_agent_protocol.cpp" \
  "$root/integrations/ggml/tests/agent_protocol_contract.cpp" \
  -o "$out"
"$out"

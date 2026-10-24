#!/usr/bin/env bash
# embedding の引きを one-hot + dot と gather で比べる（issue #240）。結果は docs/embedding-gather.md。
#
#   scripts/bench-embedding.sh                              # 語彙数 1000 / 10000 / 50000
#   NABLA_EMBED_VOCABS="1000 10000" scripts/bench-embedding.sh
#
# eager は常に測り、IREE のライブラリ（NABLA_IREE_HOME）があれば jit（local）も測る。1行1レコードの S 式。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${NABLA_LISP_DEPS:-${HOME:?NABLA_LISP_DEPS も HOME も未設定です}/.local/share/nabla/lisp-deps}"
export CL_SOURCE_REGISTRY="${REPO_ROOT}/:${DEPS_DIR}//:"

LOG="$(mktemp "${TMPDIR:-/tmp}/bench-embedding.XXXXXX")"
trap 'rm -f "$LOG"' EXIT
if sbcl --dynamic-space-size 8GB --noinform --non-interactive \
     --load "${REPO_ROOT}/scripts/bench-embedding.lisp" > "$LOG" 2>&1; then
  grep '^(:' "$LOG"
else
  grep '^(:' "$LOG" || true
  tail -40 "$LOG" >&2
  exit 1
fi

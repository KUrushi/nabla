#!/usr/bin/env bash
# 並列ローダ（issue #286）の 1 エポックの時間を、ワーカー数と先読みを変えて測る。結果は docs/data-loader-parallel.md。
#
#   scripts/bench-data-loader.sh                                  # ワーカー 0 1 2 4 8、レコード 1024 個
#   NABLA_LOADER_WORKERS="0 4" NABLA_LOADER_RECORDS=256 scripts/bench-data-loader.sh
#
# ほかの環境変数: NABLA_LOADER_STEP_MS（呼び出し側のバッチごとの計算、既定 20）、NABLA_LOADER_REPS（既定 3）。
# IREE も PJRT も使わない（nabla/data だけ）。1行1レコードの S 式。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${NABLA_LISP_DEPS:-${HOME:?NABLA_LISP_DEPS も HOME も未設定です}/.local/share/nabla/lisp-deps}"
export CL_SOURCE_REGISTRY="${REPO_ROOT}/:${DEPS_DIR}//:"

LOG="$(mktemp "${TMPDIR:-/tmp}/bench-data-loader.XXXXXX")"
trap 'rm -f "$LOG"' EXIT
if sbcl --noinform --non-interactive --load "${REPO_ROOT}/scripts/bench-data-loader.lisp" > "$LOG" 2>&1; then
  grep '^(:' "$LOG" || true
else
  grep '^(:' "$LOG" || true
  tail -40 "$LOG" >&2
  exit 1
fi

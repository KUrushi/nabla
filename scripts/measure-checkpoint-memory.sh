#!/usr/bin/env bash
# save-checkpoint / load-checkpoint のピークメモリを測る（issue #244）。結果は docs/checkpoint-memory.md。
#
#   scripts/measure-checkpoint-memory.sh                  # 64/256/1024 MiB × few/mixed × save/load、dynamic-space-size は 1GB と 8GB
#   NABLA_CKPT_SIZES="64 256" NABLA_CKPT_HEAPS="8GB" scripts/measure-checkpoint-memory.sh
#
# ケースごとに別の SBCL プロセスで測る（RSS の最大値がケース間で混ざらないように）。
# 1行1ケースの S 式を標準出力に出す。書き出したファイルは終わりに消す。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${NABLA_LISP_DEPS:-${HOME:?NABLA_LISP_DEPS も HOME も未設定です}/.local/share/nabla/lisp-deps}"
export CL_SOURCE_REGISTRY="${REPO_ROOT}/:${DEPS_DIR}//:"

SIZES="${NABLA_CKPT_SIZES:-64 256 1024}"
HEAPS="${NABLA_CKPT_HEAPS:-1GB 8GB}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ckpt-measure.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

run() { # heap op mb shape file
  NABLA_CKPT_OP="$2" NABLA_CKPT_MB="$3" NABLA_CKPT_SHAPE="$4" NABLA_CKPT_FILE="$5" \
    sbcl --dynamic-space-size "$1" --noinform --disable-ldb --lose-on-corruption --non-interactive \
      --eval '(require :asdf)' --load "${REPO_ROOT}/scripts/measure-checkpoint-memory.lisp" \
    > "$WORK/log" 2>&1 || true
  local op shape
  op="$(echo "$2" | tr a-z A-Z)"; shape="$(echo "$4" | tr a-z A-Z)"
  # 1行の結果が出ていなければ、Lisp の外（GC 中のヒープ枯渇など）で落ちている
  grep '^(:OP' "$WORK/log" || echo "(:OP :$op :MB $3 :SHAPE :$shape :DYNAMIC-SPACE-REQUEST \"$1\" :STATUS :CRASHED :REASON \"$(grep -m1 -i 'exhausted\|fatal' "$WORK/log" | tr -d '"')\")"
}

echo ";; $(sbcl --version), $(uname -sm), dynamic-space-size: ${HEAPS}"
for heap in $HEAPS; do
  for mb in $SIZES; do
    for shape in few mixed; do
      file="$WORK/ckpt-$mb-$shape.safetensors"
      # 読む側の測定用のファイルは、十分大きなヒープで別に作る
      NABLA_CKPT_OP=save NABLA_CKPT_MB="$mb" NABLA_CKPT_SHAPE="$shape" NABLA_CKPT_FILE="$file" \
        sbcl --dynamic-space-size 8GB --noinform --non-interactive \
          --eval '(require :asdf)' --load "${REPO_ROOT}/scripts/measure-checkpoint-memory.lisp" >/dev/null 2>&1
      run "$heap" save "$mb" "$shape" "$WORK/out-$mb-$shape.safetensors"
      run "$heap" load "$mb" "$shape" "$file"
      rm -f "$WORK/out-$mb-$shape.safetensors"
    done
  done
done

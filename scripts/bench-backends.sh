#!/usr/bin/env bash
# IREE (local) と PJRT (XLA CPU) で 2層 MLP のコンパイル時間と学習ステップ時間を測る（issue #89）。
#
#   scripts/bench-backends.sh                       # iree-local と pjrt-cpu、small,medium,large
#   scripts/bench-backends.sh --cuda                # iree-cuda と pjrt-cuda も（GPU が無ければ「未測定」）
#   scripts/bench-backends.sh --configs small --steps 50 --reps 1
#   scripts/bench-backends.sh --out bench.sexp      # 生のレコード（1行1つの S 式）の保存先
#
# backend ごとに別の SBCL プロセスで測り（初回コストと LLVM のシグナル処理が互いに
# 影響しないように）、レコードを --out（既定は標準エラーに出さず一時ファイル）に集めてから
# 人間向けの表を標準出力に書く。測定条件は表の前に出す。
# 環境変数: NABLA_IREE_HOME / NABLA_PJRT_HOME（run-tests.sh と同じ）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${NABLA_LISP_DEPS:-${HOME:?NABLA_LISP_DEPS も HOME も未設定です}/.local/share/nabla/lisp-deps}"
export CL_SOURCE_REGISTRY="${REPO_ROOT}/:${DEPS_DIR}//:"

backends="iree-local,pjrt-cpu"
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cuda) backends="iree-local,pjrt-cpu,iree-cuda,pjrt-cuda" ;;
    --backends) backends="$2"; shift ;;
    --configs) export NABLA_BENCH_CONFIGS="$2"; shift ;;
    --steps) export NABLA_BENCH_STEPS="$2"; shift ;;
    --warmup) export NABLA_BENCH_WARMUP="$2"; shift ;;
    --reps) export NABLA_BENCH_REPS="$2"; shift ;;
    --out) out="$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "$out" ]; then
  out="$(mktemp)"
  trap 'rm -f "$out"' EXIT
fi
: > "$out"

IFS=',' read -ra list <<< "$backends"
for backend in "${list[@]}"; do
  echo "measuring $backend ..." >&2
  NABLA_BENCH_BACKEND="$backend" sbcl --noinform --non-interactive \
    --eval '(require :asdf)' \
    --load "${REPO_ROOT}/scripts/bench-backends.lisp" \
    --eval '(nabla-bench:main)' >> "$out"
done

sbcl --noinform --non-interactive \
  --eval '(require :asdf)' \
  --load "${REPO_ROOT}/scripts/bench-backends.lisp" \
  --eval "(let ((records (with-open-file (s \"$out\") (nabla-bench:parse-records s))))
            (format t \"~&## 測定条件~%~%\")
            (dolist (r (remove-if-not (lambda (r) (eq (getf r :kind) :env)) records))
              (format t \"- ~A:~{ ~(~A~)=~A~^;~}~%\" (getf r :backend)
                      (loop for (k v) on r by #'cddr unless (member k '(:kind :backend)) append (list k v))))
            (write-string (nabla-bench:records-table records)))"

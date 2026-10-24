#!/usr/bin/env bash
# 本体に制御構造（cond*）がある scan の呼び出し時間を、外側の長さ L と ys の大きさを変えて測る（issue #233）。
# 背景と variant の意味は scripts/bench-scan-ys.lisp の冒頭。IREE（local）が要る。
#
#   scripts/bench-scan-ys.sh                                   # L=16,32,64,128 × W=4096,65536 の plain と cond
#   scripts/bench-scan-ys.sh --lengths 32,64 --widths 4096 --reps 5
#   scripts/bench-scan-ys.sh --variants plain,cond,scripts/scan-ys-candidates/index-barrier.mlir.in
#   scripts/bench-scan-ys.sh --out scan-ys.sexp                # 生のレコードの保存先
#   scripts/bench-scan-ys.sh --render-only --out scan-ys.sexp  # 測らずに、保存済みのレコードから表だけ作る
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${NABLA_LISP_DEPS:-${HOME:?NABLA_LISP_DEPS も HOME も未設定です}/.local/share/nabla/lisp-deps}"
export CL_SOURCE_REGISTRY="${REPO_ROOT}/:${DEPS_DIR}//:"

out=""
render_only=0
while [ $# -gt 0 ]; do
  case "$1" in
    --lengths) export NABLA_BENCH_SCAN_LENGTHS="$2"; shift ;;
    --widths) export NABLA_BENCH_SCAN_WIDTHS="$2"; shift ;;
    --variants) export NABLA_BENCH_SCAN_VARIANTS="$2"; shift ;;
    --reps) export NABLA_BENCH_SCAN_REPS="$2"; shift ;;
    --out) out="$2"; shift ;;
    --render-only) render_only=1 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# --variants の相対パスはリポジトリ直下を基準に解決する
if [ -n "${NABLA_BENCH_SCAN_VARIANTS:-}" ]; then
  resolved=""
  IFS=',' read -ra vs <<< "$NABLA_BENCH_SCAN_VARIANTS"
  for v in "${vs[@]}"; do
    case "$v" in
      plain|cond|/*) ;;
      *) [ -e "${REPO_ROOT}/$v" ] && v="${REPO_ROOT}/$v" ;;
    esac
    resolved="${resolved:+$resolved,}$v"
  done
  export NABLA_BENCH_SCAN_VARIANTS="$resolved"
fi

if [ "$render_only" -eq 1 ] && [ -z "$out" ]; then
  echo "--render-only には --out でレコードのファイルを渡す" >&2; exit 2
fi
if [ -z "$out" ]; then
  out="$(mktemp)"
  trap 'rm -f "$out"' EXIT
fi
if [ "$render_only" -eq 0 ]; then
  NABLA_BENCH_BACKEND=iree-local sbcl --noinform --non-interactive \
    --eval '(require :asdf)' \
    --load "${REPO_ROOT}/scripts/bench-backends.lisp" \
    --load "${REPO_ROOT}/scripts/bench-scan-ys.lisp" \
    --eval '(nabla-bench::scan-ys-main)' > "$out"
fi

sbcl --noinform --non-interactive \
  --eval '(require :asdf)' \
  --load "${REPO_ROOT}/scripts/bench-backends.lisp" \
  --eval "(let ((records (with-open-file (s \"$out\") (nabla-bench:parse-records s))))
            (format t \"~&## 測定条件~%~%\")
            (dolist (r (remove-if-not (lambda (r) (eq (getf r :kind) :env)) records))
              (format t \"- ~A:~{ ~(~A~)=~A~^;~}~%\" (getf r :backend)
                      (loop for (k v) on r by #'cddr unless (member k '(:kind :backend)) append (list k v))))
            (write-string (nabla-bench:records-table records)))"

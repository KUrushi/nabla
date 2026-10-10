#!/usr/bin/env bash
# docs/iree-repros/ の最小再現（IREE のコンパイラのバグ。docs/iree-upstream-bugs.md）を
# 単体の iree-compile にかけ、バグが今も再現するかを報告する。IREE を上げたときに
# 回避策を外せるかの最初の確認に使う（手順は docs/iree-build.md）。
#
# 使い方:
#   scripts/check-iree-repros.sh             # 各ファイルを 10 回ずつコンパイルする
#   scripts/check-iree-repros.sh --runs 30   # 回数を変える
#
# 各行の判定:
#   bug     ファイル: 1 回でもシグナル（終了コード 128 超）で落ちれば REPRODUCED
#   control ファイル（回避策を入れた形）: 全回コンパイルできれば OK
# 終了コード: 0 = 記録どおり（バグは全部再現し、回避策の形は全部通る）、
#             1 = 記録と違う（直ったバグがある、または回避策の形が落ちた）、
#             2 = iree-compile が見つからない・引数の誤り。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NABLA_IREE_HOME="${NABLA_IREE_HOME:-${HOME}/.local/share/nabla/iree-3.11.0}"
IREE_COMPILE="${NABLA_IREE_HOME}/bin/iree-compile"
REPRO_DIR="${REPO_ROOT}/docs/iree-repros"
RUNS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || { echo "error: --runs needs a positive integer" >&2; exit 2; }
      RUNS="$2"; shift 2 ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "${IREE_COMPILE}" ]]; then
  echo "error: iree-compile not found at ${IREE_COMPILE}. Set NABLA_IREE_HOME." >&2
  exit 2
fi

# 種別 ファイル名（docs/iree-upstream-bugs.md の表と同じ順）
CASES=(
  "bug     dot-general-k0.mlir"
  "bug     while-constant-carry.mlir"
  "control while-constant-carry-barrier.mlir"
  "bug     while-i1-carry-returned.mlir"
  "control while-i1-carry-not-returned.mlir"
)

"${IREE_COMPILE}" --version | sed -n '2p'
status=0
for entry in "${CASES[@]}"; do
  read -r kind file <<<"${entry}"
  crashes=0; failures=0; codes=""
  for ((i = 0; i < RUNS; i++)); do
    code=0
    # set -e のもとで落ちる子プロセスを数えるため、終了コードは || で拾う。
    # { } 2>/dev/null は bash 自身の「Segmentation fault」の通知も黙らせる
    { "${IREE_COMPILE}" --iree-hal-target-device=local \
        --iree-hal-local-target-device-backends=llvm-cpu \
        "${REPRO_DIR}/${file}" -o /dev/null >/dev/null; } 2>/dev/null || code=$?
    if ((code > 128)); then crashes=$((crashes + 1)); fi
    if ((code != 0)); then failures=$((failures + 1)); fi
    case " ${codes} " in *" ${code} "*) ;; *) codes="${codes:+${codes} }${code}" ;; esac
  done
  if [[ "${kind}" == bug ]]; then
    if ((crashes > 0)); then verdict="REPRODUCED"; else verdict="NOT-REPRODUCED"; status=1; fi
  else
    if ((failures == 0)); then verdict="OK"; else verdict="FAILED"; status=1; fi
  fi
  printf '%-8s %-40s %-15s crashes=%d/%d exit-codes=%s\n' \
    "${kind}" "${file}" "${verdict}" "${crashes}" "${RUNS}" "${codes}"
done
exit "${status}"

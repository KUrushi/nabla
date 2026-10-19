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
#   run-bug / run-control: コンパイルした vmfb を iree-run-module で local-task device で
#           実行する版（コンパイルは通り、実行時に落ちるバグ。入力の個数ぶんの --input を
#           自動で作る）。判定は bug / control と同じ
# 終了コード: 0 = 記録どおり（バグは全部再現し、回避策の形は全部通る）、
#             1 = 記録と違う（直ったバグがある、または回避策の形が落ちた）、
#             2 = iree-compile が見つからない・引数の誤り。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NABLA_IREE_HOME="${NABLA_IREE_HOME:-${HOME}/.local/share/nabla/iree-3.11.0}"
IREE_COMPILE="${NABLA_IREE_HOME}/bin/iree-compile"
IREE_RUN="${NABLA_IREE_HOME}/bin/iree-run-module"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
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

if [[ ! -x "${IREE_COMPILE}" || ! -x "${IREE_RUN}" ]]; then
  echo "error: iree-compile / iree-run-module not found under ${NABLA_IREE_HOME}/bin. Set NABLA_IREE_HOME." >&2
  exit 2
fi

# 種別 ファイル名（docs/iree-upstream-bugs.md の表と同じ順）
CASES=(
  "bug     dot-general-k0.mlir"
  "bug     while-constant-carry.mlir"
  "control while-constant-carry-barrier.mlir"
  "bug     while-i1-carry-returned.mlir"
  "control while-i1-carry-not-returned.mlir"
  "run-bug     local-task-162-inputs.mlir"
  "run-control local-task-161-inputs.mlir"
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
    if [[ "${kind}" == run-* ]]; then
      # コンパイルは通り、実行時（local-task device）に落ちるバグ。main の入力の個数ぶん
      # --input=3xf32=1,2,3 を作って実行する
      { "${IREE_COMPILE}" --iree-hal-target-device=local \
          --iree-hal-local-target-device-backends=llvm-cpu \
          "${REPRO_DIR}/${file}" -o "${WORK_DIR}/m.vmfb" >/dev/null; } 2>/dev/null || { echo "error: ${file} did not compile" >&2; exit 2; }
      inputs=$(grep -m1 'func.func @main' "${REPRO_DIR}/${file}" | grep -o '%a[0-9]*:' | wc -l)
      input_args=(); for ((j = 0; j < inputs; j++)); do input_args+=(--input=3xf32=1,2,3); done
      { "${IREE_RUN}" --device=local-task --module="${WORK_DIR}/m.vmfb" --function=main \
          "${input_args[@]}" >/dev/null; } 2>/dev/null || code=$?
    else
      { "${IREE_COMPILE}" --iree-hal-target-device=local \
          --iree-hal-local-target-device-backends=llvm-cpu \
          "${REPRO_DIR}/${file}" -o /dev/null >/dev/null; } 2>/dev/null || code=$?
    fi
    if ((code > 128)); then crashes=$((crashes + 1)); fi
    if ((code != 0)); then failures=$((failures + 1)); fi
    case " ${codes} " in *" ${code} "*) ;; *) codes="${codes:+${codes} }${code}" ;; esac
  done
  if [[ "${kind}" == bug || "${kind}" == run-bug ]]; then
    if ((crashes > 0)); then verdict="REPRODUCED"; else verdict="NOT-REPRODUCED"; status=1; fi
  else
    if ((failures == 0)); then verdict="OK"; else verdict="FAILED"; status=1; fi
  fi
  printf '%-11s %-40s %-15s crashes=%d/%d exit-codes=%s\n' \
    "${kind}" "${file}" "${verdict}" "${crashes}" "${RUNS}" "${codes}"
done
exit "${status}"

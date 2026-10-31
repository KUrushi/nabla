#!/usr/bin/env bash
# docs/iree-repros/ の最小再現（IREE 3.11 のバグ 1〜9 と制限 10。docs/iree-upstream-bugs.md）を
# 単体の iree-compile / iree-run-module にかけ、バグが今も再現するかを報告する。IREE を上げたときに
# 回避策を外せるかの最初の確認に使う（手順は docs/iree-build.md）。
#
# 使い方:
#   scripts/check-iree-repros.sh             # 各ファイルを 10 回ずつコンパイル（run-* は実行も）する
#   scripts/check-iree-repros.sh --runs 30   # 回数を変える
#   scripts/check-iree-repros.sh --race-runs 5000   # race-* の実行回数（既定 4000。1 回あたり 0.4〜2% でしか起きない（負荷のもとでの実測。docs/iree-upstream-bugs.md の 7））
#   scripts/check-iree-repros.sh --candidates --race-runs 20000   # 追加で race-candidate（下）も実行する
#   scripts/check-iree-repros.sh --case nested-scan-ys-joint-barrier-race.mlir --race-runs 20000
#       # 名前を挙げた行だけを実行する（繰り返して複数を選べる。表の順に実行する。race-candidate の
#       # ファイルも --candidates なしで選べる）。ケースごとに別のジョブで走らせる CI 用
#       # （.github/workflows/iree-race-check.yml、issue #233）
#
# 各行の判定:
#   bug     ファイル: 1 回でもシグナル（終了コード 128 超）で落ちれば REPRODUCED
#   control ファイル（回避策を入れた形）: 全回コンパイルできれば OK
#   error-bug ファイル: クラッシュではなく通常のコンパイルエラー（診断を出して終了コード 1）で
#           落ちるバグ。1 回でも終了コードが 0 でなければ REPRODUCED（対照は control）。
#           group-norm-grad-multiplier.mlir は 1 回あたり約 17% でしか落ちないので、--runs 20 以上で見る
#   run-bug / run-control: コンパイルした vmfb を iree-run-module で local-task device で
#           実行する版（コンパイルは通り、実行時に落ちるバグ。入力の個数ぶんの --input を
#           自動で作る）。判定は bug / control と同じ
#   fail-bug / fail-control: 実行時にエラー（ref is null など。クラッシュではなく終了コード 1）で
#           落ちるバグ。vmfb を iree-run-module で local-task と local-sync の両方で実行する。
#           入力はファイルの `// INPUT: --input=...` の行（最初の入力は pred 用で、bug は 1.0 の
#           else の枝で落ちる。control は 0.0 と 1.0 の両方で通る）。fail-bug は 1 回でも
#           終了コードが 0 でなければ REPRODUCED、fail-control は全部 0 なら OK
#   limit-bug / limit-control: ロード時に制限（binding count N > 32。docs/iree-upstream-bugs.md の 10）で
#           失敗する形（終了コード 1）。nabla の回避策のフラグを付けずに iree-compile した vmfb を
#           local-task と local-sync の両方で実行する（入力は main の引数の個数ぶんの --input=3xf32=1,2,3）。
#           limit-bug は 1 回でも終了コードが 0 でなければ REPRODUCED（出力に "binding count" が
#           無い失敗は別の原因なので終了コード 2 で止める）、limit-control は全部 0 なら OK
#   race-bug / race-control: 実行時に、まれに結果が壊れるバグ（クラッシュではない）。vmfb を
#           local-sync で 1 回実行した結果を正とし、local-task で RACE_RUNS 回実行して
#           出力が違った回数を数える。入力はファイルの `// INPUT: --input=...` の行。
#           負荷のもとで起きるので、このスクリプトが CPU の数だけ `yes > /dev/null` を
#           起動し（終了時にこのスクリプトが自分で止める）、かけながら実行する。
#           race-bug は 1 回でも違えば REPRODUCED、race-control は 0 回なら OK
#   race-candidate: nabla の emit の改善の候補（issue #233）の形。race-bug と同じ測り方で、
#           違った回数を数えるだけ。記録（上流の不具合の再現）ではないので判定も終了コードも
#           変えない。--candidates を付けたときだけ実行する。0 回でも安全の証明にはならない
#           （Rosetta の VM では race-bug 自体がほとんど再現しない（while 版 0/200、if 版 1/200）。x86 ネイティブの負荷のもとで、
#           race-bug が REPRODUCED になる環境で、十分な回数 0 になるかを見る）。
#           候補を足すには docs/iree-repros/ に race 用の形（`// INPUT:` の行つき）を置き、
#           CANDIDATE_CASES に1行足す
# 終了コード: 0 = 記録どおり（バグは全部再現し、回避策の形は全部通る）、
#             1 = 記録と違う（直ったバグがある、または回避策の形が落ちた）、
#             2 = iree-compile が見つからない・引数の誤り。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NABLA_IREE_HOME="${NABLA_IREE_HOME:-${HOME}/.local/share/nabla/iree-3.11.0}"
IREE_COMPILE="${NABLA_IREE_HOME}/bin/iree-compile"
IREE_RUN="${NABLA_IREE_HOME}/bin/iree-run-module"
WORK_DIR="$(mktemp -d)"
STRESS_PIDS=()
# ${a[@]+...} は空の配列を set -u のもとで展開するため（bash 4.3 以前は空の "${a[@]}" を unbound とみなす）
trap 'for pid in ${STRESS_PIDS[@]+"${STRESS_PIDS[@]}"}; do kill "${pid}" 2>/dev/null || true; done; rm -rf "${WORK_DIR}"' EXIT
REPRO_DIR="${REPO_ROOT}/docs/iree-repros"
RUNS=10
RACE_RUNS=4000
CANDIDATES=0
ONLY=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || { echo "error: --runs needs a positive integer" >&2; exit 2; }
      RUNS="$2"; shift 2 ;;
    --race-runs)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || { echo "error: --race-runs needs a positive integer" >&2; exit 2; }
      RACE_RUNS="$2"; shift 2 ;;
    --candidates) CANDIDATES=1; shift ;;
    --case)
      [[ $# -ge 2 && -n "$2" ]] || { echo "error: --case needs a file name" >&2; exit 2; }
      ONLY+=("$2"); shift 2 ;;
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
  "error-bug    conv-convert-operand-type.mlir"
  "control      conv-convert-operand-type-barrier.mlir"
  "error-bug    conv-depthwise-collapse-shape.mlir"
  "control      conv-depthwise-collapse-shape-barrier.mlir"
  "error-bug    group-norm-grad-multiplier.mlir"
  "control      group-norm-grad-normalize-first.mlir"
  "run-bug     local-task-162-inputs.mlir"
  "run-control local-task-161-inputs.mlir"
  "fail-bug     if-passthrough-operands.mlir"
  "fail-control if-passthrough-operands-select.mlir"
  "limit-bug     elementwise-chain-32-inputs.mlir"
  "limit-control elementwise-chain-31-inputs.mlir"
  "race-bug     nested-scan-ys-barrier-race.mlir"
  "race-control nested-scan-ys-no-barrier.mlir"
  "race-bug     nested-scan-ys-if-race.mlir"
  "race-control nested-scan-ys-if-no-barrier.mlir"
)

# --candidates のときだけ実行する（docs/iree-upstream-bugs.md の 7、issue #233）
CANDIDATE_CASES=(
  "race-candidate nested-scan-ys-index-barrier-race.mlir"
  "race-candidate nested-scan-ys-joint-barrier-race.mlir"
  "race-candidate nested-scan-ys-if-index-barrier-race.mlir"
  "race-candidate nested-scan-ys-if-joint-barrier-race.mlir"
)
if ((CANDIDATES)); then CASES+=("${CANDIDATE_CASES[@]}"); fi

# --case を付けたときは、全部の行（CANDIDATE_CASES を含む）から名前を挙げた行だけを残す
if ((${#ONLY[@]} > 0)); then
  ALL_CASES=("${CASES[@]}")
  if ((!CANDIDATES)); then ALL_CASES+=("${CANDIDATE_CASES[@]}"); fi
  ALL_FILES=" "
  for entry in "${ALL_CASES[@]}"; do read -r kind file <<<"${entry}"; ALL_FILES+="${file} "; done
  for name in "${ONLY[@]}"; do
    [[ "${ALL_FILES}" == *" ${name} "* ]] || { echo "error: --case: no such case: ${name}" >&2; exit 2; }
  done
  ONLY_FILES=" ${ONLY[*]} "
  CASES=()
  for entry in "${ALL_CASES[@]}"; do
    read -r kind file <<<"${entry}"
    if [[ "${ONLY_FILES}" == *" ${file} "* ]]; then CASES+=("${entry}"); fi
  done
fi

# race-* は負荷のもとでしか起きないので、CPU の数だけ busy loop を起動する（trap で止める）
start_stress() {
  local n; n=$(nproc)
  for ((k = 0; k < n; k++)); do yes >/dev/null & STRESS_PIDS+=($!); done
}

# race_run ファイル: 出力が local-sync と違った local-task の実行の回数を返す
race_run() {
  local file="$1" args=() line
  while IFS= read -r line; do args+=("${line#// INPUT: }"); done < <(grep '^// INPUT: ' "${REPRO_DIR}/${file}")
  { "${IREE_COMPILE}" --iree-hal-target-device=local \
      --iree-hal-local-target-device-backends=llvm-cpu \
      "${REPRO_DIR}/${file}" -o "${WORK_DIR}/r.vmfb" >/dev/null; } 2>/dev/null || { echo "error: ${file} did not compile" >&2; exit 2; }
  local ref cur bad=0 i
  ref=$("${IREE_RUN}" --device=local-sync --module="${WORK_DIR}/r.vmfb" --function=main "${args[@]}" 2>&1)
  for ((i = 0; i < RACE_RUNS; i++)); do
    cur=$("${IREE_RUN}" --device=local-task --module="${WORK_DIR}/r.vmfb" --function=main "${args[@]}" 2>&1) || true
    [[ "${cur}" == "${ref}" ]] || bad=$((bad + 1))
  done
  echo "${bad}"
}

# fail_run ファイル pred: 入力の最初を pred に差し替え、local-task と local-sync の両方で実行して、
# 終了コードが 0 でなかった実行の回数を返す
fail_run() {
  local file="$1" pred="$2" args=() line first=1 bad=0 dev
  while IFS= read -r line; do
    if ((first)); then args+=("--input=f32=${pred}"); first=0; else args+=("${line#// INPUT: }"); fi
  done < <(grep '^// INPUT: ' "${REPRO_DIR}/${file}")
  { "${IREE_COMPILE}" --iree-hal-target-device=local \
      --iree-hal-local-target-device-backends=llvm-cpu \
      "${REPRO_DIR}/${file}" -o "${WORK_DIR}/f.vmfb" >/dev/null; } 2>/dev/null || { echo "error: ${file} did not compile" >&2; exit 2; }
  for dev in local-task local-sync; do
    "${IREE_RUN}" --device="${dev}" --module="${WORK_DIR}/f.vmfb" --function=main "${args[@]}" >/dev/null 2>&1 || bad=$((bad + 1))
  done
  echo "${bad}"
}

# limit_run ファイル: local-task と local-sync の両方で実行し、終了コードが 0 でなかった実行の回数を返す。
# bug の側は、失敗した実行の出力に "binding count" が無ければ（別の理由で落ちた）コンパイルエラー扱いで止める
limit_run() {
  local file="$1" kind="$2" bad=0 dev out inputs j input_args=()
  { "${IREE_COMPILE}" --iree-hal-target-device=local \
      --iree-hal-local-target-device-backends=llvm-cpu \
      "${REPRO_DIR}/${file}" -o "${WORK_DIR}/l.vmfb" >/dev/null; } 2>/dev/null || { echo "error: ${file} did not compile" >&2; exit 2; }
  inputs=$(grep -m1 'func.func @main' "${REPRO_DIR}/${file}" | grep -o '%a[0-9]*:' | wc -l)
  for ((j = 0; j < inputs; j++)); do input_args+=(--input=3xf32=1,2,3); done
  for dev in local-task local-sync; do
    if ! out=$("${IREE_RUN}" --device="${dev}" --module="${WORK_DIR}/l.vmfb" --function=main "${input_args[@]}" 2>&1); then
      bad=$((bad + 1))
      if [[ "${kind}" == limit-bug && "${out}" != *"binding count"* ]]; then
        echo "error: ${file} failed on ${dev} for a reason other than the binding count" >&2; exit 2
      fi
    fi
  done
  echo "${bad}"
}

"${IREE_COMPILE}" --version | sed -n '2p'
status=0
start_stress
for entry in "${CASES[@]}"; do
  read -r kind file <<<"${entry}"
  if [[ "${kind}" == race-* ]]; then
    # race_run の中の `exit 2`（コンパイル失敗）は $(...) のサブシェルを抜けるだけだが、
    # 代入の終了コードが 2 になり、set -e がこのスクリプトを止める（意図した挙動）
    bad=$(race_run "${file}")
    if [[ "${kind}" == race-candidate ]]; then
      verdict="NO-MISMATCH"; if ((bad > 0)); then verdict="RACES"; fi
    elif [[ "${kind}" == race-bug ]]; then
      if ((bad > 0)); then verdict="REPRODUCED"; else verdict="NOT-REPRODUCED"; status=1; fi
    else
      if ((bad == 0)); then verdict="OK"; else verdict="FAILED"; status=1; fi
    fi
    printf '%-12s %-40s %-15s mismatches=%d/%d\n' "${kind}" "${file}" "${verdict}" "${bad}" "${RACE_RUNS}"
    continue
  fi
  if [[ "${kind}" == limit-* ]]; then
    bad=$(limit_run "${file}" "${kind}")
    if [[ "${kind}" == limit-bug ]]; then
      if ((bad > 0)); then verdict="REPRODUCED"; else verdict="NOT-REPRODUCED"; status=1; fi
    else
      if ((bad == 0)); then verdict="OK"; else verdict="FAILED"; status=1; fi
    fi
    printf '%-13s %-40s %-15s failing-runs=%d\n' "${kind}" "${file}" "${verdict}" "${bad}"
    continue
  fi
  if [[ "${kind}" == fail-* ]]; then
    # else の枝（pred = 1.0）で落ちるか。control は then の枝（0.0）も確かめる
    bad=$(fail_run "${file}" 1.0)
    if [[ "${kind}" == fail-control ]]; then bad=$((bad + $(fail_run "${file}" 0.0))); fi
    if [[ "${kind}" == fail-bug ]]; then
      if ((bad > 0)); then verdict="REPRODUCED"; else verdict="NOT-REPRODUCED"; status=1; fi
    else
      if ((bad == 0)); then verdict="OK"; else verdict="FAILED"; status=1; fi
    fi
    printf '%-12s %-40s %-15s failing-runs=%d\n' "${kind}" "${file}" "${verdict}" "${bad}"
    continue
  fi
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
  elif [[ "${kind}" == error-bug ]]; then
    if ((failures > 0)); then verdict="REPRODUCED"; else verdict="NOT-REPRODUCED"; status=1; fi
  else
    if ((failures == 0)); then verdict="OK"; else verdict="FAILED"; status=1; fi
  fi
  printf '%-11s %-40s %-15s crashes=%d/%d exit-codes=%s\n' \
    "${kind}" "${file}" "${verdict}" "${crashes}" "${RUNS}" "${codes}"
done
exit "${status}"

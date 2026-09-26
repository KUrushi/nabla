#!/usr/bin/env bash
# scripts/build-iree.sh でビルドした iree-compile / iree-run-module を使い、
# tests/fixtures/stablehlo/matmul.mlir を実際にコンパイル・実行して結果を確かめる。
#
# 使い方:
#   scripts/verify-iree.sh          # llvm-cpu のみ
#   scripts/verify-iree.sh --cuda   # llvm-cpu に加えて cuda でも確かめる (nvidia-smi が必要)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NABLA_IREE_HOME="${NABLA_IREE_HOME:-${HOME}/.local/share/nabla/iree-3.11.0}"
MATMUL_MLIR="${REPO_ROOT}/tests/fixtures/stablehlo/matmul.mlir"

IREE_COMPILE="${NABLA_IREE_HOME}/bin/iree-compile"
IREE_RUN_MODULE="${NABLA_IREE_HOME}/bin/iree-run-module"

WITH_CUDA=0
for arg in "$@"; do
  case "${arg}" in
    --cuda) WITH_CUDA=1 ;;
    *)
      echo "error: unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

log() {
  echo "[verify-iree] $*"
}

if [[ ! -x "${IREE_COMPILE}" ]]; then
  echo "error: iree-compile not found at ${IREE_COMPILE}. Run scripts/build-iree.sh first." >&2
  exit 1
fi
if [[ ! -x "${IREE_RUN_MODULE}" ]]; then
  echo "error: iree-run-module not found at ${IREE_RUN_MODULE}. Run scripts/build-iree.sh first." >&2
  exit 1
fi
if [[ ! -f "${MATMUL_MLIR}" ]]; then
  echo "error: fixture not found: ${MATMUL_MLIR}" >&2
  exit 1
fi

# a = [[1,2,3],[4,5,6]] (2x3), b = [[7,8],[9,10],[11,12]] (3x2)
# expected a @ b = [[58,64],[139,154]] (2x2), see the comment in matmul.mlir.
INPUT_A="2x3xf32=1,2,3,4,5,6"
INPUT_B="3x2xf32=7,8,9,10,11,12"
EXPECTED_VALUES=(58 64 139 154)

check_output() {
  local backend="$1"
  local output="$2"
  # iree-run-module prints "result[0]: hal.buffer_view\n2x2xf32=[58 64][139
  # 154]" (its pretty-printer may split the tensor data across lines, hence
  # the newline flattening). We only look at the data after the buffer
  # view's final "=" (the shape prefix, e.g. "2x2xf32", is before it), then
  # require the parsed numbers to be EXACTLY [58 64 139 154] in order.
  #
  # A plain \b58\b-style regex on the whole output is not enough: word
  # boundaries sit around the full token "58.5" too, so \b58\b still matches
  # its "58" prefix and a fractional wrong result (e.g. "58.5 64 139 154")
  # would wrongly pass. Parsing the exact number list rules that out.
  local flattened data_part numbers expected
  flattened="$(tr '\n' ' ' <<<"${output}")"
  # Taking everything after the LAST "=" assumes exactly one result buffer
  # in the output, which holds for main's single tensor return in
  # matmul.mlir. If a future fixture ever returns more than one result,
  # this would need to isolate result[0]'s own line instead.
  data_part="${flattened##*=}"
  numbers="$(grep -oP -- '-?[0-9]+(\.[0-9]+)?' <<<"${data_part}")"
  expected="$(printf '%s\n' "${EXPECTED_VALUES[@]}")"
  if [[ "${numbers}" == "${expected}" ]]; then
    log "${backend}: OK, output matches expected [[58 64][139 154]] exactly"
  else
    echo "error: ${backend}: output did not contain exactly the expected values [58 64 139 154] in order" >&2
    echo "  got: ${output}" >&2
    exit 1
  fi
}

run_backend() {
  local backend="$1"       # llvm-cpu | cuda
  local hal_target="$2"    # local | cuda
  local device="$3"        # local-task | cuda
  local vmfb="$4"

  log "compiling for ${backend} -> ${vmfb}"
  if [[ "${backend}" == "llvm-cpu" ]]; then
    # --iree-llvmcpu-target-cpu=host targets this machine's actual CPU
    # instead of the generic baseline, which silences iree-compile's
    # "using default configuration" warning.
    "${IREE_COMPILE}" \
      --iree-hal-target-device=local \
      --iree-hal-local-target-device-backends=llvm-cpu \
      --iree-llvmcpu-target-cpu=host \
      "${MATMUL_MLIR}" -o "${vmfb}"
  else
    "${IREE_COMPILE}" \
      --iree-hal-target-device=cuda \
      "${MATMUL_MLIR}" -o "${vmfb}"
  fi

  log "running ${backend} module with iree-run-module"
  local output
  output="$("${IREE_RUN_MODULE}" \
    --device="${device}" \
    --module="${vmfb}" \
    --function=main \
    --input="${INPUT_A}" \
    --input="${INPUT_B}")"
  check_output "${backend}" "${output}"
}

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nabla-matmul.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

run_backend "llvm-cpu" "local" "local-task" "${WORK_DIR}/nabla-matmul-llvm-cpu.vmfb"

if [[ "${WITH_CUDA}" -eq 1 ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "skipping cuda check: nvidia-smi not found (no GPU in this environment)"
  else
    run_backend "cuda" "cuda" "cuda" "${WORK_DIR}/nabla-matmul-cuda.vmfb"
  fi
else
  log "skipping cuda check (pass --cuda to enable; requires an IREE build with"
  log "  IREE_TARGET_BACKEND_CUDA=ON, which in turn needs a CUDA toolkit"
  log "  installed or network access to NVIDIA's redistributable package"
  log "  index at build-configure time; see docs/iree-build.md)"
fi

log "all checks passed"

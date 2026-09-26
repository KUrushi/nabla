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
  # Require the four expected values to appear, IN ORDER, each bounded by
  # non-digit characters, so this can't be satisfied by a substring match
  # (e.g. "158") or by the right numbers appearing in the wrong positions
  # (e.g. a transposed result). We flatten newlines first since the values
  # may be split across lines by iree-run-module's pretty-printer.
  local flattened
  flattened="$(tr '\n' ' ' <<<"${output}")"
  # \b (zero-width word boundary) avoids the classic bug of consuming the
  # delimiter between two required numbers, which would otherwise make the
  # match fail for legitimate output or, worse, silently succeed on the
  # wrong subset of characters.
  local order_regex='\b58\b.*\b64\b.*\b139\b.*\b154\b'
  if grep -qP "${order_regex}" <<<"${flattened}"; then
    log "${backend}: OK, output matches expected [[58 64][139 154]] in order"
  else
    echo "error: ${backend}: output did not contain expected values [58 64 139 154] in order" >&2
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
    "${IREE_COMPILE}" \
      --iree-hal-target-device=local \
      --iree-hal-local-target-device-backends=llvm-cpu \
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
  log "skipping cuda check (pass --cuda to enable; requires an IREE build with IREE_TARGET_BACKEND_CUDA=ON)"
fi

log "all checks passed"

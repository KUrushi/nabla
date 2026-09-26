#!/usr/bin/env bash
# IREE を third_party/iree.lock で固定したコミットからソースビルドし、
# NABLA_IREE_HOME にインストールする。
#
# 使い方:
#   scripts/build-iree.sh                  # CPU (llvm-cpu) のみ
#   scripts/build-iree.sh --cuda           # CUDA も有効化
#   scripts/build-iree.sh --configure-only # cmake configure までで止める
#
# 冪等性: 既にロックしたコミットのチェックアウトがあれば再利用し、
# 既存のビルドディレクトリがあれば cmake の再設定だけ行って増分ビルドする。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${REPO_ROOT}/third_party/iree.lock"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "error: lock file not found: ${LOCK_FILE}" >&2
  exit 1
fi

# third_party/iree.lock を読む (key=value の3行)。
LOCK_REPO=""
LOCK_TAG=""
LOCK_COMMIT=""
while IFS='=' read -r key value; do
  case "${key}" in
    repo) LOCK_REPO="${value}" ;;
    tag) LOCK_TAG="${value}" ;;
    commit) LOCK_COMMIT="${value}" ;;
  esac
done < <(grep -E '^(repo|tag|commit)=' "${LOCK_FILE}")

if [[ -z "${LOCK_REPO}" || -z "${LOCK_TAG}" || -z "${LOCK_COMMIT}" ]]; then
  echo "error: failed to parse ${LOCK_FILE}" >&2
  exit 1
fi

NABLA_IREE_SRC="${NABLA_IREE_SRC:-/home/user/iree-src}"
NABLA_IREE_BUILD="${NABLA_IREE_BUILD:-/home/user/iree-build}"
NABLA_IREE_HOME="${NABLA_IREE_HOME:-${HOME}/.local/share/nabla/iree-3.11.0}"
NABLA_IREE_JOBS="${NABLA_IREE_JOBS:-$(nproc)}"
NABLA_IREE_CUDA="${NABLA_IREE_CUDA:-0}"

CONFIGURE_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --cuda) NABLA_IREE_CUDA=1 ;;
    --configure-only) CONFIGURE_ONLY=1 ;;
    *)
      echo "error: unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

log() {
  echo "[build-iree] $(date -u '+%Y-%m-%dT%H:%M:%SZ') $*"
}

SECONDS=0

log "locked to ${LOCK_REPO} tag=${LOCK_TAG} commit=${LOCK_COMMIT}"
log "NABLA_IREE_SRC=${NABLA_IREE_SRC}"
log "NABLA_IREE_BUILD=${NABLA_IREE_BUILD}"
log "NABLA_IREE_HOME=${NABLA_IREE_HOME}"
log "NABLA_IREE_JOBS=${NABLA_IREE_JOBS}"
log "NABLA_IREE_CUDA=${NABLA_IREE_CUDA}"
log "disk usage before: $(df -h "${NABLA_IREE_SRC%/*}" 2>/dev/null | tail -1 || true)"

# --- 1. ソースの取得 (再利用 or clone) --------------------------------------

need_clone=1
if [[ -d "${NABLA_IREE_SRC}/.git" || -f "${NABLA_IREE_SRC}/.git" ]]; then
  current_commit="$(git -C "${NABLA_IREE_SRC}" rev-parse HEAD 2>/dev/null || true)"
  if [[ "${current_commit}" == "${LOCK_COMMIT}" ]]; then
    log "reusing existing checkout at ${NABLA_IREE_SRC} (HEAD=${current_commit})"
    need_clone=0
  else
    log "existing checkout at ${NABLA_IREE_SRC} is at ${current_commit}, not ${LOCK_COMMIT}; will re-checkout"
    need_clone=0
    git -C "${NABLA_IREE_SRC}" fetch --depth 1 origin "${LOCK_COMMIT}"
    if ! git -C "${NABLA_IREE_SRC}" checkout "${LOCK_COMMIT}"; then
      echo "error: could not check out ${LOCK_COMMIT} in ${NABLA_IREE_SRC}." >&2
      echo "       this usually means there are local modifications or untracked" >&2
      echo "       files that would be overwritten. Inspect with:" >&2
      echo "         git -C ${NABLA_IREE_SRC} status" >&2
      exit 1
    fi
  fi
fi

if [[ "${need_clone}" -eq 1 ]]; then
  log "cloning ${LOCK_REPO} at ${LOCK_TAG} into ${NABLA_IREE_SRC}"
  git clone --branch "${LOCK_TAG}" --depth 1 "${LOCK_REPO}" "${NABLA_IREE_SRC}"
  git -C "${NABLA_IREE_SRC}" fetch --depth 1 origin "${LOCK_COMMIT}"
  git -C "${NABLA_IREE_SRC}" checkout "${LOCK_COMMIT}"
fi

# 必要なサブモジュールだけを浅く取得する。
SUBMODULES=(
  third_party/llvm-project
  third_party/stablehlo
  third_party/flatcc
  third_party/printf
  third_party/benchmark
  third_party/googletest
  third_party/musl
)
log "initializing submodules: ${SUBMODULES[*]}"
git -C "${NABLA_IREE_SRC}" submodule update --init --depth 1 "${SUBMODULES[@]}"

# musl は upstream で heads/main を指しており depth 1 の submodule update が
# 目的のコミットに届かないことがある。その場合は記録済みの SHA を直接取得する。
musl_pin="3f701faace7addc75d16dea8a6cd769fa5b3f260"
musl_dir="${NABLA_IREE_SRC}/third_party/musl"
musl_head="$(git -C "${musl_dir}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${musl_head}" != "${musl_pin}" ]]; then
  log "musl submodule HEAD (${musl_head}) != pinned ${musl_pin}; fetching pinned commit"
  git -C "${musl_dir}" fetch --depth 1 origin "${musl_pin}" || true
  git -C "${musl_dir}" checkout "${musl_pin}" || \
    log "warning: could not pin musl to ${musl_pin}; continuing with ${musl_head}"
fi

# --- 2. cmake configure ------------------------------------------------------

mkdir -p "${NABLA_IREE_BUILD}"

CMAKE_ARGS=(
  -G Ninja
  -S "${NABLA_IREE_SRC}"
  -B "${NABLA_IREE_BUILD}"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=clang
  -DCMAKE_CXX_COMPILER=clang++
  -DIREE_ENABLE_LLD=ON
  -DIREE_ENABLE_THIN_ARCHIVES=ON
  -DIREE_ENABLE_WERROR_FLAG=OFF
  -DIREE_ENABLE_ASSERTIONS=OFF
  # CFFI は dlopen したライブラリのシンボルテーブルを見て関数を探すので、
  # デフォルトの -fvisibility=hidden のままではシムから何も見えなくなる。
  -DIREE_VISIBILITY_HIDDEN=OFF
  # We deliberately init only the submodules this build needs (see
  # SUBMODULES below), not the ones for backends we don't build (ROCm,
  # Vulkan, WebGPU, Torch, tracing, ...). IREE's own submodule-init check
  # does not know about --runtime_only vs. --compiler-with-only-these-
  # backends, so it would otherwise fail on those intentionally-uninitialized
  # submodules.
  -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF
  -DIREE_BUILD_COMPILER=ON
  -DIREE_BUILD_TESTS=OFF
  -DIREE_BUILD_SAMPLES=OFF
  -DIREE_BUILD_PYTHON_BINDINGS=OFF
  -DIREE_BUILD_BINDINGS_TFLITE=OFF
  -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON
  -DIREE_TARGET_BACKEND_VMVX=OFF
  -DIREE_HAL_DRIVER_DEFAULTS=OFF
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON
  -DIREE_INPUT_STABLEHLO=ON
  -DIREE_INPUT_TORCH=OFF
  -DIREE_INPUT_TOSA=OFF
)

if [[ "${NABLA_IREE_CUDA}" == "1" ]]; then
  log "CUDA target/driver requested (NABLA_IREE_CUDA=1 / --cuda)"
  CMAKE_ARGS+=(
    -DIREE_TARGET_BACKEND_CUDA=ON
    -DIREE_HAL_DRIVER_CUDA=ON
  )
else
  CMAKE_ARGS+=(
    -DIREE_TARGET_BACKEND_CUDA=OFF
    -DIREE_HAL_DRIVER_CUDA=OFF
  )
fi

log "running cmake configure"
cmake "${CMAKE_ARGS[@]}"

log "configured option values:"
cmake -L -N "${NABLA_IREE_BUILD}" | grep -E 'IREE_(TARGET_BACKEND|HAL_DRIVER|INPUT|BUILD|ENABLE)' || true

if [[ "${CONFIGURE_ONLY}" -eq 1 ]]; then
  log "--configure-only: stopping after cmake configure"
  log "elapsed: ${SECONDS}s"
  exit 0
fi

# --- 3. ビルド ---------------------------------------------------------------

BUILD_TARGETS=(
  iree-compile
  iree-run-module
  iree_compiler_API_SharedImpl
  iree_runtime_unified
)
log "building targets: ${BUILD_TARGETS[*]} (-j ${NABLA_IREE_JOBS})"
cmake --build "${NABLA_IREE_BUILD}" --target "${BUILD_TARGETS[@]}" -j "${NABLA_IREE_JOBS}"

# --- 4. ランタイムの共有ライブラリを作る (静的アーカイブを --whole-archive で包む) ----

mkdir -p "${NABLA_IREE_HOME}/lib" "${NABLA_IREE_HOME}/bin" "${NABLA_IREE_HOME}/include"

RUNTIME_STATIC_LIB="${NABLA_IREE_BUILD}/runtime/src/iree/runtime/libiree_runtime_unified.a"
if [[ ! -f "${RUNTIME_STATIC_LIB}" ]]; then
  echo "error: expected static library not found: ${RUNTIME_STATIC_LIB}" >&2
  exit 1
fi

# iree_cc_unified_library (iree_runtime_unified の生成元) は third_party の
# 依存 (flatcc, printf) をアーカイブに含めず INTERFACE_IREE_TRANSITIVE_OBJECT_LIBS
# として記録するだけなので、シムをリンクするときは別途これらの静的ライブラリを
# 明示的に渡す必要がある。ビルドツリー内を検索して見つける (パスはターゲット名
# から生成されるが、サブディレクトリ構成はバージョン間で動きうるため固定しない)。
find_static_lib() {
  local name="$1"
  local -a matches
  mapfile -t matches < <(find "${NABLA_IREE_BUILD}" -name "${name}" | sort)
  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "error: required static library not found in build tree: ${name}" >&2
    exit 1
  fi
  if [[ "${#matches[@]}" -gt 1 ]]; then
    log "warning: multiple candidates for ${name} found in ${NABLA_IREE_BUILD}, using the first (sorted by path): ${matches[*]}"
  fi
  echo "${matches[0]}"
}

FLATCC_STATIC_LIB="$(find_static_lib 'libflatcc_parsing.a')"
PRINTF_STATIC_LIB="$(find_static_lib 'libprintf_printf.a')"
log "found flatcc runtime static lib: ${FLATCC_STATIC_LIB}"
log "found printf static lib: ${PRINTF_STATIC_LIB}"

SHIM_SO="${NABLA_IREE_HOME}/lib/libnabla_iree_runtime.so"
log "linking shim shared library: ${SHIM_SO}"
if ! clang -shared -fPIC -fuse-ld=lld \
  -o "${SHIM_SO}" \
  -Wl,--whole-archive "${RUNTIME_STATIC_LIB}" "${FLATCC_STATIC_LIB}" "${PRINTF_STATIC_LIB}" -Wl,--no-whole-archive \
  -Wl,--no-undefined \
  -lpthread -ldl -lm; then
  echo "error: linking ${SHIM_SO} failed with -Wl,--no-undefined." >&2
  echo "       re-run with: clang -shared -fPIC -fuse-ld=lld -o ${SHIM_SO} -Wl,--whole-archive ${RUNTIME_STATIC_LIB} ${FLATCC_STATIC_LIB} ${PRINTF_STATIC_LIB} -Wl,--no-whole-archive -lpthread -ldl -lm" >&2
  echo "       then inspect undefined symbols with: nm -u ${RUNTIME_STATIC_LIB}" >&2
  exit 1
fi

log "verifying exported symbols in ${SHIM_SO}"
for sym in iree_runtime_instance_create iree_hal_driver_registry_default; do
  # nm -D lists the dynamic symbol table, which includes both symbols this
  # library DEFINES (type letter other than U, e.g. T/W/t/w) and symbols it
  # merely references and expects to find elsewhere (type U, undefined). Only
  # a non-U match means the symbol is actually exported from this .so.
  if ! nm -D "${SHIM_SO}" | grep -qE "^[0-9a-fA-F]+ [^U[:space:]] ${sym}\$"; then
    echo "error: expected exported symbol not found (or undefined) in ${SHIM_SO}: ${sym}" >&2
    nm -D "${SHIM_SO}" | grep -i "${sym%%_*}" || true
    exit 1
  fi
done
log "confirmed exports: iree_runtime_instance_create, iree_hal_driver_registry_default"

# --- 5. インストール ---------------------------------------------------------

log "installing lib/, bin/, include/ into ${NABLA_IREE_HOME}"

if ! cp -a "${NABLA_IREE_BUILD}"/lib*/libIREECompiler.so* "${NABLA_IREE_HOME}/lib/" 2>/dev/null; then
  found_compiler_so="$(find "${NABLA_IREE_BUILD}" -maxdepth 4 -name 'libIREECompiler.so*' -print -quit)"
  if [[ -z "${found_compiler_so}" ]]; then
    echo "error: libIREECompiler.so not found under ${NABLA_IREE_BUILD}" >&2
    exit 1
  fi
  find "${NABLA_IREE_BUILD}" -maxdepth 4 -name 'libIREECompiler.so*' -exec cp -a {} "${NABLA_IREE_HOME}/lib/" \;
fi
# IREE_ENABLE_THIN_ARCHIVES=ON なので libiree_runtime_unified.a はビルド
# ディレクトリ内のオブジェクトファイルへの相対パス参照でしかなく、単独で
# コピーしても壊れたアーカイブにしかならない。デバッグ時はビルドツリーの
# ものをそのまま参照すること。

cp -a "${NABLA_IREE_BUILD}/tools/iree-compile" "${NABLA_IREE_HOME}/bin/"
cp -a "${NABLA_IREE_BUILD}/tools/iree-run-module" "${NABLA_IREE_HOME}/bin/"

# コンパイラの C API ヘッダ一式。
mkdir -p "${NABLA_IREE_HOME}/include/iree/compiler"
cp -a "${NABLA_IREE_SRC}/compiler/bindings/c/iree/compiler/embedding_api.h" \
      "${NABLA_IREE_SRC}/compiler/bindings/c/iree/compiler/api_support.h" \
      "${NABLA_IREE_SRC}/compiler/bindings/c/iree/compiler/loader.h" \
      "${NABLA_IREE_SRC}/compiler/bindings/c/iree/compiler/mlir_interop.h" \
      "${NABLA_IREE_HOME}/include/iree/compiler/"

# ランタイムのヘッダツリー全体 (ソース側 + ビルドで生成されたスキーマ)。
# cpio はこの環境にインストールされていないため使わない。ディレクトリ構造を
# 保ったまま .h だけをコピーするのに find -exec install -D を使う。
mkdir -p "${NABLA_IREE_HOME}/include/iree"
(cd "${NABLA_IREE_SRC}/runtime/src/iree" && \
  find . -name '*.h' -exec install -D -m 0644 '{}' "${NABLA_IREE_HOME}/include/iree/{}" \;)
if [[ -d "${NABLA_IREE_BUILD}/runtime/src/iree" ]]; then
  (cd "${NABLA_IREE_BUILD}/runtime/src/iree" && \
    find . -name '*.h' -exec install -D -m 0644 '{}' "${NABLA_IREE_HOME}/include/iree/{}" \;)
fi

log "install complete: ${NABLA_IREE_HOME}"
ls -la "${NABLA_IREE_HOME}/lib" "${NABLA_IREE_HOME}/bin"

log "disk usage after: $(df -h "${NABLA_IREE_BUILD}" | tail -1)"
log "build dir size: $(du -sh "${NABLA_IREE_BUILD}" 2>/dev/null | cut -f1)"
log "elapsed: ${SECONDS}s"

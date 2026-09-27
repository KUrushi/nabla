#!/usr/bin/env bash
# IREE を third_party/iree.lock で固定したコミットからビルド・インストールし、
# NABLA_IREE_HOME に配置する。
#
# コンパイラ (libIREECompiler.so, iree-compile) は2通りの入手方法がある
# (--compiler / NABLA_IREE_COMPILER で選ぶ。既定は wheel):
#
#   wheel  (既定) third_party/iree.lock に記録した PyPI ホイール
#          (iree-base-compiler) をダウンロードして中身を取り出す。
#          このホイールはロックしたコミットと同じビルドから作られている
#          ことを `iree-compile --version` の出力で確認する。
#   source ロックしたコミットからコンパイラをフルソースビルドする。
#          このマシン (4 コア / 15 GB RAM) では ninja が 1 時間で
#          7085 ステップ中 5674 までしか進まず、実用的な時間では終わらない
#          (詳細は docs/iree-build.md)。CI や、より強力なマシンでのみ使う。
#
# ランタイム (libnabla_iree_runtime.so) はどちらのモードでも常にロックした
# コミットからソースビルドする (PyPI のコンパイラ用ホイールに C ランタイムは
# 含まれていないため)。ランタイムだけのビルドは ninja で数秒で終わる。
#
# 使い方:
#   scripts/build-iree.sh                     # wheel コンパイラ + ソースランタイム (CPU)
#   scripts/build-iree.sh --compiler=source   # コンパイラもソースビルド
#   scripts/build-iree.sh --cuda              # CUDA も有効化
#   scripts/build-iree.sh --configure-only    # cmake configure までで止める
#
# 注意:
# - wheel モード (既定) は third_party/iree.lock の manylinux cp311 ホイール
#   しか使わないため x86_64 Linux 専用。他のプラットフォーム (macOS, aarch64
#   など) では --compiler=source を使う (詳細は docs/iree-build.md)
# - --cuda は CUDA toolkit がインストール済みか、cmake configure 時に
#   NVIDIA の redistributable パッケージ索引にネットワークで到達できる必要が
#   ある。どちらもない環境では configure が失敗する (詳細は docs/iree-build.md)
#
# 環境変数 (すべて未設定なら既定値を使う):
#   NABLA_IREE_HOME     インストール先。既定 ~/.local/share/nabla/iree-3.11.0
#   NABLA_IREE_SRC      IREE のソースチェックアウト先。既定は
#                       ${XDG_CACHE_HOME:-~/.cache}/nabla/iree-src
#   NABLA_IREE_BUILD    cmake のビルドディレクトリ。既定は
#                       .../nabla/iree-build-runtime (wheel) または
#                       .../nabla/iree-build (source)
#   NABLA_IREE_WHEEL_DIR ダウンロードしたホイールの保存先。既定は
#                       .../nabla/iree-wheel
#   NABLA_IREE_COMPILER wheel|source。--compiler と同じ
#   NABLA_IREE_CUDA     1 で CUDA を有効化。--cuda と同じ
#   NABLA_IREE_JOBS     ninja の並列数。既定は nproc
#
# 冪等性: 既にロックしたコミットのチェックアウトがあれば再利用し、
# 既存のビルドディレクトリがあれば cmake の再設定だけ行って増分ビルドする。
# wheel はダウンロード先に既に正しい sha256 のファイルがあれば再ダウンロードせず、
# 展開先に既に中身があれば再展開しない。libIREECompiler.so は、インストール先に
# 既に同じサイズ・mtime のファイルがあれば再コピーしない (copy_if_changed)。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${REPO_ROOT}/third_party/iree.lock"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "error: lock file not found: ${LOCK_FILE}" >&2
  exit 1
fi

# third_party/iree.lock を読む (key=value)。
LOCK_REPO=""
LOCK_TAG=""
LOCK_COMMIT=""
LOCK_WHEEL_NAME=""
LOCK_WHEEL_VERSION=""
LOCK_WHEEL_SHA256=""
while IFS='=' read -r key value; do
  case "${key}" in
    repo) LOCK_REPO="${value}" ;;
    tag) LOCK_TAG="${value}" ;;
    commit) LOCK_COMMIT="${value}" ;;
    wheel_name) LOCK_WHEEL_NAME="${value}" ;;
    wheel_version) LOCK_WHEEL_VERSION="${value}" ;;
    wheel_sha256) LOCK_WHEEL_SHA256="${value}" ;;
  esac
done < <(grep -E '^(repo|tag|commit|wheel_name|wheel_version|wheel_sha256)=' "${LOCK_FILE}")

if [[ -z "${LOCK_REPO}" || -z "${LOCK_TAG}" || -z "${LOCK_COMMIT}" ]]; then
  echo "error: failed to parse repo/tag/commit from ${LOCK_FILE}" >&2
  exit 1
fi

NABLA_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}/nabla"
NABLA_IREE_SRC="${NABLA_IREE_SRC:-${NABLA_CACHE_HOME}/iree-src}"
NABLA_IREE_HOME="${NABLA_IREE_HOME:-${HOME}/.local/share/nabla/iree-3.11.0}"
NABLA_IREE_JOBS="${NABLA_IREE_JOBS:-$(nproc)}"
NABLA_IREE_CUDA="${NABLA_IREE_CUDA:-0}"
NABLA_IREE_COMPILER="${NABLA_IREE_COMPILER:-wheel}"

CONFIGURE_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --cuda) NABLA_IREE_CUDA=1 ;;
    --configure-only) CONFIGURE_ONLY=1 ;;
    --compiler=*) NABLA_IREE_COMPILER="${arg#*=}" ;;
    *)
      echo "error: unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ "${NABLA_IREE_COMPILER}" != "wheel" && "${NABLA_IREE_COMPILER}" != "source" ]]; then
  echo "error: --compiler must be 'wheel' or 'source', got: ${NABLA_IREE_COMPILER}" >&2
  exit 1
fi

if [[ "${NABLA_IREE_COMPILER}" == "wheel" ]]; then
  if [[ -z "${LOCK_WHEEL_NAME}" || -z "${LOCK_WHEEL_VERSION}" || -z "${LOCK_WHEEL_SHA256}" ]]; then
    echo "error: --compiler=wheel requires wheel_name/wheel_version/wheel_sha256 in ${LOCK_FILE}" >&2
    exit 1
  fi
  NABLA_IREE_BUILD="${NABLA_IREE_BUILD:-${NABLA_CACHE_HOME}/iree-build-runtime}"
else
  NABLA_IREE_BUILD="${NABLA_IREE_BUILD:-${NABLA_CACHE_HOME}/iree-build}"
fi
NABLA_IREE_WHEEL_DIR="${NABLA_IREE_WHEEL_DIR:-${NABLA_CACHE_HOME}/iree-wheel}"

log() {
  echo "[build-iree] $(date -u '+%Y-%m-%dT%H:%M:%SZ') $*"
}

# libIREECompiler.so is ~337 MB (wheel mode) or similarly large when built
# from source, so re-copying it on every run (even when the source hasn't
# changed) wastes real time and disk I/O. Skip the copy when the destination
# already has the same size and mtime as the source (cp -a preserves both,
# so this correctly detects "already installed from this exact source file"
# across runs without reading the whole file).
# GNU stat (Linux, this environment) and BSD stat (macOS) take different
# flags for the same "size and mtime" query, so try GNU's first and fall
# back to BSD's.
stat_size_mtime() {
  stat -c '%s %Y' "$1" 2>/dev/null || stat -f '%z %m' "$1"
}

copy_if_changed() {
  local src="$1" dst="$2"
  if [[ -f "${dst}" ]]; then
    if [[ "$(stat_size_mtime "${src}")" == "$(stat_size_mtime "${dst}")" ]]; then
      log "skipping copy, unchanged (size+mtime match): ${dst}"
      return 0
    fi
  fi
  cp -a "${src}" "${dst}"
}

SECONDS=0

log "locked to ${LOCK_REPO} tag=${LOCK_TAG} commit=${LOCK_COMMIT}"
log "NABLA_IREE_COMPILER=${NABLA_IREE_COMPILER}"
log "NABLA_IREE_SRC=${NABLA_IREE_SRC}"
log "NABLA_IREE_BUILD=${NABLA_IREE_BUILD}"
log "NABLA_IREE_HOME=${NABLA_IREE_HOME}"
if [[ "${NABLA_IREE_COMPILER}" == "wheel" ]]; then
  log "NABLA_IREE_WHEEL_DIR=${NABLA_IREE_WHEEL_DIR}"
  log "locked wheel: ${LOCK_WHEEL_NAME}==${LOCK_WHEEL_VERSION} sha256=${LOCK_WHEEL_SHA256}"
fi
log "NABLA_IREE_JOBS=${NABLA_IREE_JOBS}"
log "NABLA_IREE_CUDA=${NABLA_IREE_CUDA}"
log "disk usage before: $(df -h "${NABLA_IREE_SRC%/*}" 2>/dev/null | tail -1 || true)"

# --- 1. ソースの取得 (再利用 or clone) --------------------------------------
#
# --compiler=wheel でも、コンパイラの C API ヘッダ (embedding_api.h 等、ホイールには
# 含まれていない) とランタイムのソースは必要なので、常にチェックアウトする。

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
  # SUBMODULES above), not the ones for backends we don't build (ROCm,
  # Vulkan, WebGPU, Torch, tracing, ...). IREE's own submodule-init check
  # does not know about --runtime_only vs. --compiler-with-only-these-
  # backends, so it would otherwise fail on those intentionally-uninitialized
  # submodules.
  -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF
  -DIREE_BUILD_TESTS=OFF
  -DIREE_BUILD_SAMPLES=OFF
  -DIREE_BUILD_PYTHON_BINDINGS=OFF
  -DIREE_BUILD_BINDINGS_TFLITE=OFF
  -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF
  -DIREE_HAL_DRIVER_DEFAULTS=OFF
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON
)

if [[ "${NABLA_IREE_COMPILER}" == "wheel" ]]; then
  # コンパイラは PyPI ホイールから取るので、このビルドはランタイムだけを
  # 作る。IREE_BUILD_COMPILER=OFF にすると IREE_TARGET_BACKEND_* /
  # IREE_INPUT_* は cmake_dependent_option により自動的に OFF になるので、
  # ここでは指定しない (CMakeLists.txt の該当行は docs/iree-build.md 参照)。
  CMAKE_ARGS+=(-DIREE_BUILD_COMPILER=OFF)
else
  CMAKE_ARGS+=(
    -DIREE_BUILD_COMPILER=ON
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON
    -DIREE_TARGET_BACKEND_VMVX=OFF
    -DIREE_INPUT_STABLEHLO=ON
    -DIREE_INPUT_TORCH=OFF
    -DIREE_INPUT_TOSA=OFF
  )
fi

if [[ "${NABLA_IREE_CUDA}" == "1" ]]; then
  log "CUDA target/driver requested (NABLA_IREE_CUDA=1 / --cuda)"
  CMAKE_ARGS+=(-DIREE_HAL_DRIVER_CUDA=ON)
  if [[ "${NABLA_IREE_COMPILER}" == "source" ]]; then
    CMAKE_ARGS+=(-DIREE_TARGET_BACKEND_CUDA=ON)
  fi
else
  CMAKE_ARGS+=(-DIREE_HAL_DRIVER_CUDA=OFF)
  if [[ "${NABLA_IREE_COMPILER}" == "source" ]]; then
    CMAKE_ARGS+=(-DIREE_TARGET_BACKEND_CUDA=OFF)
  fi
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

# --- 3. コンパイラ (wheel モード): 取得と検証 (fail fast) --------------------
#
# ninja によるランタイムのビルドに入る前に、wheel のダウンロードと sha256
# 検証を済ませておく。悪いホイール (壊れたダウンロード、third_party/iree.lock
# の更新忘れなど) はここで即座に失敗させ、ランタイムのビルドを始めてから
# 半端な状態を残すことを避ける。
if [[ "${NABLA_IREE_COMPILER}" == "wheel" ]]; then
  mkdir -p "${NABLA_IREE_HOME}/lib" "${NABLA_IREE_HOME}/bin"

  # third_party/iree.lock に記録した PyPI ホイールから libIREECompiler.so と
  # iree-compile を取り出す。このホイールは同じコミット (LOCK_COMMIT) から
  # ビルドされているはずで、それを iree-compile --version の出力で確認する。
  WHEEL_GLOB="${LOCK_WHEEL_NAME//-/_}-${LOCK_WHEEL_VERSION}-*.whl"
  mkdir -p "${NABLA_IREE_WHEEL_DIR}"

  WHEEL_PATH="$(find "${NABLA_IREE_WHEEL_DIR}" -maxdepth 1 -name "${WHEEL_GLOB}" -print -quit)"
  if [[ -n "${WHEEL_PATH}" ]]; then
    log "found cached wheel: ${WHEEL_PATH}"
  else
    log "downloading ${LOCK_WHEEL_NAME}==${LOCK_WHEEL_VERSION} into ${NABLA_IREE_WHEEL_DIR}"
    # Pin the exact wheel tag (matching the one recorded in
    # third_party/iree.lock) rather than letting pip pick whatever matches
    # the local interpreter. Without this, running the script on a
    # different Python/platform would either find no compatible wheel or
    # silently fetch a different one, and only fail much later with a
    # confusing sha256 mismatch.
    #
    # This wheel is only published for manylinux x86_64 (see
    # docs/iree-build.md); on other platforms (macOS, aarch64, ...) use
    # --compiler=source instead.
    pip download --no-deps --only-binary=:all: \
      --python-version 311 --implementation cp --abi cp311 \
      --platform manylinux_2_28_x86_64 \
      "${LOCK_WHEEL_NAME}==${LOCK_WHEEL_VERSION}" -d "${NABLA_IREE_WHEEL_DIR}"
    WHEEL_PATH="$(find "${NABLA_IREE_WHEEL_DIR}" -maxdepth 1 -name "${WHEEL_GLOB}" -print -quit)"
    if [[ -z "${WHEEL_PATH}" ]]; then
      echo "error: pip download did not produce a file matching ${WHEEL_GLOB} in ${NABLA_IREE_WHEEL_DIR}" >&2
      exit 1
    fi
  fi

  log "verifying sha256 of ${WHEEL_PATH}"
  ACTUAL_SHA256="$(sha256sum "${WHEEL_PATH}" | cut -d' ' -f1)"
  if [[ "${ACTUAL_SHA256}" != "${LOCK_WHEEL_SHA256}" ]]; then
    # Remove the bad file instead of leaving it in the cache: otherwise the
    # "found cached wheel" branch above would keep finding this same file
    # (matched by filename glob alone) and failing the same way on every
    # subsequent run, requiring a human to clear NABLA_IREE_WHEEL_DIR by hand.
    rm -f "${WHEEL_PATH}"
    echo "error: sha256 mismatch for ${WHEEL_PATH} (removed; re-run to re-download)" >&2
    echo "       expected (third_party/iree.lock): ${LOCK_WHEEL_SHA256}" >&2
    echo "       actual:                           ${ACTUAL_SHA256}" >&2
    exit 1
  fi

  EXTRACT_DIR="${NABLA_IREE_WHEEL_DIR}/extracted-${LOCK_WHEEL_VERSION}"
  WHEEL_COMPILER_SO="${EXTRACT_DIR}/iree/compiler/_mlir_libs/libIREECompiler.so"
  WHEEL_IREE_COMPILE="${EXTRACT_DIR}/iree/compiler/_mlir_libs/iree-compile"
  WHEEL_IREE_LLD="${EXTRACT_DIR}/iree/compiler/_mlir_libs/iree-lld"
  if [[ ! -f "${WHEEL_COMPILER_SO}" || ! -x "${WHEEL_IREE_COMPILE}" ]]; then
    log "extracting ${WHEEL_PATH} into ${EXTRACT_DIR}"
    rm -rf "${EXTRACT_DIR}"
    python3 -m zipfile -e "${WHEEL_PATH}" "${EXTRACT_DIR}"
  else
    log "reusing existing extraction at ${EXTRACT_DIR}"
  fi
  # python's zipfile module does not restore the Unix executable bit stored
  # in the archive, so iree-compile / iree-lld come out as plain -rw-r--r--.
  # Do this unconditionally (not only in the freshly-extracted branch above):
  # an extraction cached from a version of this script that did not yet know
  # about iree-lld would otherwise be "reused" with iree-lld still
  # non-executable, since the branch above only re-extracts based on
  # iree-compile's executable bit.
  if [[ -f "${WHEEL_IREE_COMPILE}" ]]; then
    chmod +x "${WHEEL_IREE_COMPILE}"
  fi
  if [[ -f "${WHEEL_IREE_LLD}" ]]; then
    chmod +x "${WHEEL_IREE_LLD}"
  fi
  if [[ ! -f "${WHEEL_COMPILER_SO}" ]]; then
    echo "error: ${WHEEL_COMPILER_SO} not found after extracting ${WHEEL_PATH}" >&2
    exit 1
  fi
  if [[ ! -x "${WHEEL_IREE_COMPILE}" ]]; then
    echo "error: ${WHEEL_IREE_COMPILE} not found (or not executable) after extracting ${WHEEL_PATH}" >&2
    exit 1
  fi
  if [[ ! -x "${WHEEL_IREE_LLD}" ]]; then
    echo "error: ${WHEEL_IREE_LLD} not found (or not executable) after extracting ${WHEEL_PATH}" >&2
    exit 1
  fi

  log "installing compiler from wheel into ${NABLA_IREE_HOME}"
  copy_if_changed "${WHEEL_COMPILER_SO}" "${NABLA_IREE_HOME}/lib/libIREECompiler.so"
  cp -a "${WHEEL_IREE_COMPILE}" "${NABLA_IREE_HOME}/bin/iree-compile"
  cp -a "${WHEEL_IREE_LLD}" "${NABLA_IREE_HOME}/bin/iree-lld"
  # iree-compile / iree-lld の RUNPATH は $ORIGIN で、同じディレクトリの
  # libIREECompiler.so を NEEDS しているので、bin/ にもシンボリックリンクを
  # 置く (実体は lib/ に置いたものを指す)。llvm-cpu バックエンドは常に外部の
  # リンカをサブプロセスで起動してオブジェクトをリンクするため (このビルドに
  # インプロセスのリンカは無い)、system の /usr/bin/lld ではなくロックした
  # IREE コミットと同じビルドの lld を使わせる必要がある
  # (--iree-llvmcpu-embedded-linker-path、呼び出し側は scripts/verify-iree.sh
  # と #20 のビルド層を参照)。
  ln -sf ../lib/libIREECompiler.so "${NABLA_IREE_HOME}/bin/libIREECompiler.so"

  log "verifying iree-lld runs and resolves libIREECompiler.so via its RUNPATH"
  IREE_LLD_VERSION_OUTPUT="$("${NABLA_IREE_HOME}/bin/iree-lld" -flavor gnu --version)"
  log "iree-lld: ${IREE_LLD_VERSION_OUTPUT}"

  log "verifying iree-compile --version reports the locked commit"
  VERSION_OUTPUT="$("${NABLA_IREE_HOME}/bin/iree-compile" --version)"
  if [[ "${VERSION_OUTPUT}" != *"${LOCK_COMMIT}"* ]]; then
    echo "error: iree-compile --version does not mention the locked commit ${LOCK_COMMIT}:" >&2
    echo "${VERSION_OUTPUT}" >&2
    exit 1
  fi
  log "confirmed: iree-compile reports commit ${LOCK_COMMIT}"
fi

# --- 4. ビルド ---------------------------------------------------------------

BUILD_TARGETS=(iree-run-module iree_runtime_unified)
if [[ "${NABLA_IREE_COMPILER}" == "source" ]]; then
  # iree-lld: llvm-cpu バックエンドは CPU 実行ファイルをリンクするのに
  # 常に外部のリンカをサブプロセスで起動する (このビルドにインプロセスの
  # リンカは無い) ので、ロックしたコミットと同じビルドの lld
  # (system の /usr/bin/lld ではなく) を --iree-llvmcpu-embedded-linker-path
  # で渡せるようにインストールしておく。
  BUILD_TARGETS+=(iree-compile iree_compiler_API_SharedImpl iree-lld)
fi
log "building targets: ${BUILD_TARGETS[*]} (-j ${NABLA_IREE_JOBS})"
cmake --build "${NABLA_IREE_BUILD}" --target "${BUILD_TARGETS[@]}" -j "${NABLA_IREE_JOBS}"

# --- 5. ランタイムの共有ライブラリを作る (静的アーカイブを --whole-archive で包む) ----

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
# Capture nm's output once instead of piping it into grep -q directly: under
# `set -o pipefail` (this script), `nm -D ... | grep -qE ...` can report
# failure even when grep DOES find a match, because grep -q exits as soon as
# it sees the match, which can send nm a SIGPIPE (killing it with a non-zero
# status) before it finishes writing the rest of its (unread) output. With
# pipefail, that non-zero producer status fails the whole pipeline
# regardless of grep's own (successful) result. Grepping over an already
# fully-captured string avoids the pipe (and the SIGPIPE) entirely.
SHIM_DYNSYMS="$(nm -D "${SHIM_SO}")"
for sym in iree_runtime_instance_create iree_hal_driver_registry_default; do
  # nm -D lists the dynamic symbol table, which includes both symbols this
  # library DEFINES (type letter other than U, e.g. T/W/t/w) and symbols it
  # merely references and expects to find elsewhere (type U, undefined). Only
  # a non-U match means the symbol is actually exported from this .so.
  if ! grep -qE "^[0-9a-fA-F]+ [^U[:space:]] ${sym}\$" <<<"${SHIM_DYNSYMS}"; then
    echo "error: expected exported symbol not found (or undefined) in ${SHIM_SO}: ${sym}" >&2
    grep -i "${sym%%_*}" <<<"${SHIM_DYNSYMS}" || true
    exit 1
  fi
done
log "confirmed exports: iree_runtime_instance_create, iree_hal_driver_registry_default"

# --- 6. コンパイラのインストール (--compiler=source のときだけ) --------------
#
# wheel モードは手順 2 で既にインストール済み。

mkdir -p "${NABLA_IREE_HOME}/lib" "${NABLA_IREE_HOME}/bin"

if [[ "${NABLA_IREE_COMPILER}" == "source" ]]; then
  # Search the build tree instead of assuming a fixed lib*/ subdirectory,
  # since that location can move between cmake/ninja versions (see comment
  # on find_static_lib above for the same reasoning).
  compiler_so_found=0
  while IFS= read -r -d '' src_so; do
    copy_if_changed "${src_so}" "${NABLA_IREE_HOME}/lib/$(basename "${src_so}")"
    compiler_so_found=1
  done < <(find "${NABLA_IREE_BUILD}" -maxdepth 4 -name 'libIREECompiler.so*' -print0)
  if [[ "${compiler_so_found}" -eq 0 ]]; then
    echo "error: libIREECompiler.so not found under ${NABLA_IREE_BUILD}" >&2
    exit 1
  fi
  # IREE_ENABLE_THIN_ARCHIVES=ON なので libiree_runtime_unified.a はビルド
  # ディレクトリ内のオブジェクトファイルへの相対パス参照でしかなく、単独で
  # コピーしても壊れたアーカイブにしかならない。デバッグ時はビルドツリーの
  # ものをそのまま参照すること。
  cp -a "${NABLA_IREE_BUILD}/tools/iree-compile" "${NABLA_IREE_HOME}/bin/"
  cp -a "${NABLA_IREE_BUILD}/tools/iree-lld" "${NABLA_IREE_HOME}/bin/"
  log "verifying iree-lld runs and resolves libIREECompiler.so via its RUNPATH"
  IREE_LLD_VERSION_OUTPUT="$("${NABLA_IREE_HOME}/bin/iree-lld" -flavor gnu --version)"
  log "iree-lld: ${IREE_LLD_VERSION_OUTPUT}"
fi

cp -a "${NABLA_IREE_BUILD}/tools/iree-run-module" "${NABLA_IREE_HOME}/bin/"

# コンパイラの C API ヘッダ一式 (wheel には含まれないので常にソースから取る)。
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

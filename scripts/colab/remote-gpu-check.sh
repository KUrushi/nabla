#!/usr/bin/env bash
# Colab の GPU VM の上で実行する（issue #12）。手元から直接は呼ばない。
# scripts/colab-gpu-check.sh がこのリポジトリを VM に送り、このスクリプトを
# バックグラウンドで起動する。
#
# やること:
#   1. SBCL と IREE のビルド道具を apt で入れ、Lisp の依存を揃える
#   2. CUDA HAL ドライバ入りで IREE ランタイムをビルドする
#   3. verify-iree.sh --cuda（iree-run-module で matmul を cuda で実行）
#   4. large テスト（tests/iree/cross-device-test.lisp ほか）を
#      NABLA_REQUIRE_CUDA=1 で実行する
#   5. local と cuda の最大誤差を実測する（measure-cross-device.lisp）
#
# 結果は $NABLA_COLAB_OUT（既定 /content/nabla-out）に、手順ごとのログと
# summary.md として残す。各手順の終了コードを summary.md に書き、
# 途中で失敗しても後の手順（測れるもの）は続ける。
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${NABLA_COLAB_OUT:-/content/nabla-out}"
mkdir -p "${OUT}"
SUMMARY="${OUT}/summary.md"
: > "${SUMMARY}"
cd "${REPO_ROOT}"

export DEBIAN_FRONTEND=noninteractive
export NABLA_IREE_HOME="${NABLA_IREE_HOME:-/root/.local/share/nabla/iree-3.11.0}"
export NABLA_LISP_DEPS="${NABLA_LISP_DEPS:-/root/.local/share/nabla/lisp-deps}"
export CL_SOURCE_REGISTRY="${REPO_ROOT}/:${NABLA_LISP_DEPS}//:"
# Colab には CUDA toolkit が /usr/local/cuda に入っている。cmake の
# FindCUDAToolkit に場所を教え、NVIDIA の索引を取りに行かせない。
if [[ -d /usr/local/cuda ]]; then
  export CUDAToolkit_ROOT=/usr/local/cuda
  export PATH="/usr/local/cuda/bin:${PATH}"
fi

step() {
  # step <名前> <コマンド...>: ログを ${OUT}/<名前>.log に残し、終了コードを記録する
  local name="$1"; shift
  local start end status
  start=$(date +%s)
  echo "=== [$(date -u +%FT%TZ)] ${name}: $*" | tee -a "${OUT}/progress.log"
  "$@" > "${OUT}/${name}.log" 2>&1
  status=$?
  end=$(date +%s)
  echo "=== ${name}: exit ${status} ($((end - start))s)" | tee -a "${OUT}/progress.log"
  echo "- ${name}: exit ${status}（$((end - start)) 秒）" >> "${SUMMARY}"
  return "${status}"
}

{
  echo "# nabla GPU check (issue #12)"
  echo
  echo "- date: $(date -u +%FT%TZ)"
  echo "- commit: $(cat "${REPO_ROOT}/.nabla-commit" 2>/dev/null || echo unknown)"
  echo "- iree.lock: $(grep -m1 -E 'commit|COMMIT' third_party/iree.lock 2>/dev/null || echo unknown)"
  echo
  echo '## GPU'
  echo
  echo '```'
  nvidia-smi -L 2>&1
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv 2>&1
  nvidia-smi 2>&1 | grep -m1 'CUDA Version' || true
  nvcc --version 2>&1 | tail -n 2 || true
  echo '```'
  echo
  echo '## 手順'
  echo
} >> "${SUMMARY}"

# Colab には pip と git が最初から入っている。apt の python3-pip で Colab の
# Python 環境（カーネルが動いている）を書き換えないよう、入れない。
step apt bash -c 'apt-get update && apt-get install -y sbcl clang lld cmake ninja-build'
step lisp-deps scripts/setup-lisp-deps.sh
if step build-iree scripts/build-iree.sh --cuda; then
  step verify-iree scripts/verify-iree.sh --cuda
  step large-tests env NABLA_TEST_SIZES=large NABLA_REQUIRE_IREE=1 NABLA_REQUIRE_CUDA=1 \
    scripts/run-tests.sh
  step measure sbcl --non-interactive --load scripts/colab/measure-cross-device.lisp
fi

{
  echo
  echo '## local と cuda の最大誤差'
  echo
  grep -E '^\|' "${OUT}/measure.log" 2>/dev/null || echo '（measure の手順が実行されなかったか失敗した。measure.log を見る）'
  echo
  echo '## large テストの結果（末尾）'
  echo
  echo '```'
  tail -n 40 "${OUT}/large-tests.log" 2>/dev/null || echo '(no log)'
  echo '```'
} >> "${SUMMARY}"

echo "done" > "${OUT}/DONE"

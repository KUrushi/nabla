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
#   6. NABLA_COLAB_BENCH=1 のとき、PJRT の CUDA プラグインを取得して
#      scripts/bench-backends.sh で GPU の backend（既定 iree-cuda と
#      pjrt-cuda。NABLA_COLAB_BENCH_BACKENDS で変える）を1つずつ測る。
#      NABLA_COLAB_BENCH_ONLY=1 なら 3〜5 を飛ばす。
#      Colab では 10 分程度で終わるよう、ステップ数などを既定より減らし
#      （NABLA_BENCH_STEPS=100 / WARMUP=10 / REPS=2）、backend ごとに
#      NABLA_COLAB_BENCH_TIMEOUT 秒（既定 600）で打ち切る
#   7. NABLA_COLAB_PROFILE=1 のとき、jit した関数の呼び出し1回を段に分けて測る
#      （issue #190、scripts/profile-call-overhead.lisp）。iree-local・iree-cuda
#      （IREE の既定の sm_60 向けと、この GPU の世代向け。issue #193）・pjrt-cuda を
#      1つずつ別プロセスで測る。
#      このときも 3〜5 を飛ばす。NABLA_COLAB_WITH_TESTS=1 なら、ベンチや内訳を
#      測るときも 3〜5 を飛ばさない
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
BENCH="${NABLA_COLAB_BENCH:-0}"
BENCH_ONLY="${NABLA_COLAB_BENCH_ONLY:-0}"
PROFILE="${NABLA_COLAB_PROFILE:-0}"
WITH_TESTS="${NABLA_COLAB_WITH_TESTS:-0}"
if step build-iree scripts/build-iree.sh --cuda; then
  if [[ "${WITH_TESTS}" == 1 || ( "${BENCH_ONLY}" != 1 && "${PROFILE}" != 1 ) ]]; then
    step verify-iree scripts/verify-iree.sh --cuda
    step large-tests env NABLA_TEST_SIZES=large NABLA_REQUIRE_IREE=1 NABLA_REQUIRE_CUDA=1 \
      scripts/run-tests.sh
    step measure sbcl --non-interactive --load scripts/colab/measure-cross-device.lisp
  fi
  if [[ "${BENCH}" == 1 || "${BENCH_ONLY}" == 1 ]]; then
    # PJRT の CUDA プラグインは CUDA のライブラリ（cuBLAS など）を実行時に
    # 探す。Colab の CUDA toolkit の場所を教える。
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    step fetch-pjrt scripts/fetch-pjrt.sh --cuda
    export NABLA_BENCH_STEPS="${NABLA_BENCH_STEPS:-100}"
    export NABLA_BENCH_WARMUP="${NABLA_BENCH_WARMUP:-10}"
    export NABLA_BENCH_REPS="${NABLA_BENCH_REPS:-2}"
    : > "${OUT}/bench.sexp"
    IFS=',' read -ra bench_backends <<< "${NABLA_COLAB_BENCH_BACKENDS:-iree-cuda,pjrt-cuda}"
    for b in "${bench_backends[@]}"; do
      # backend ごとに別の手順にして、どこまで進んだかを progress.log に残す。
      # timeout はプロセスグループごと止めるので、子の sbcl も残らない
      step "bench-${b}" timeout --kill-after=30 "${NABLA_COLAB_BENCH_TIMEOUT:-600}" \
        scripts/bench-backends.sh --backends "${b}" --out "${OUT}/bench-${b}.sexp"
      if [[ ! -s "${OUT}/bench-${b}.sexp" ]] || ! grep -q ':kind :result' "${OUT}/bench-${b}.sexp"; then
        echo "(:kind :result :backend \"${b}\" :config \"-\" :metric \"-\" :status :unmeasured :reason \"timed out or produced no records (see bench-${b}.log)\")" \
          >> "${OUT}/bench-${b}.sexp"
      fi
      cat "${OUT}/bench-${b}.sexp" >> "${OUT}/bench.sexp"
    done
    # 表は bench.log（標準出力）。backend ごとの失敗は「未測定」（理由つき）で残る
    step bench scripts/bench-backends.sh --render-only --out "${OUT}/bench.sexp"
  fi
  if [[ "${PROFILE}" == 1 ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    [[ -f "${OUT}/fetch-pjrt.log" ]] || step fetch-pjrt scripts/fetch-pjrt.sh --cuda
    : > "${OUT}/profile.sexp"
    # 「列名|backend|IREE のランタイムフラグ|cuda-arch」。ランタイムフラグを変えても
    # 速さは変わらなかった（docs/phase2-report.md §4.5）ので、既定のまま。
    # cuda-arch は nvidia-smi の compute capability から作る（T4 なら sm_75。issue #193）。
    # 空なら IREE の既定（sm_60）
    cuda_arch="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .')"
    profile_variants=(
      "iree-local|iree-local||"
      "iree-cuda|iree-cuda||"
      "iree-cuda+${cuda_arch}|iree-cuda||${cuda_arch}"
      "pjrt-cuda|pjrt-cuda||"
    )
    for v in "${profile_variants[@]}"; do
      IFS='|' read -r label backend flags arch <<< "${v}"
      step "profile-${label}" env NABLA_BENCH_BACKEND="${backend}" NABLA_PROFILE_LABEL="${label}" \
        NABLA_PROFILE_IREE_FLAGS="${flags}" NABLA_PROFILE_CUDA_ARCH="${arch}" \
        timeout --kill-after=30 "${NABLA_COLAB_PROFILE_TIMEOUT:-300}" \
        sbcl --noinform --non-interactive --eval '(require :asdf)' \
        --load scripts/bench-backends.lisp --load scripts/profile-call-overhead.lisp \
        --eval '(nabla-bench::profile-main)'
      # 標準出力（レコード）と標準エラー（進み具合）は同じログに混ざるので、レコードの行だけを拾う
      if grep -q '^(:KIND :RESULT' "${OUT}/profile-${label}.log"; then
        grep '^(:KIND' "${OUT}/profile-${label}.log" >> "${OUT}/profile.sexp"
      else
        echo "(:kind :result :backend \"${label}\" :config \"-\" :metric \"-\" :status :unmeasured :reason \"timed out or produced no records (see profile-${label}.log)\")" \
          >> "${OUT}/profile.sexp"
      fi
    done
    step profile scripts/bench-backends.sh --render-only --out "${OUT}/profile.sexp"
  fi
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
  if [[ -f "${OUT}/bench.log" ]]; then
    echo
    echo '## ベンチ（scripts/bench-backends.sh）'
    echo
    echo "ステップ ${NABLA_BENCH_STEPS:-200}、ウォームアップ ${NABLA_BENCH_WARMUP:-20}、コンパイルの繰り返し ${NABLA_BENCH_REPS:-3}"
    echo
    # 「measuring ...」などの進み具合を除き、測定条件と表だけを残す
    sed -n '/^## 測定条件/,$p' "${OUT}/bench.log"
  fi
  if [[ -f "${OUT}/profile.log" ]]; then
    echo
    echo '## 呼び出し1回の内訳（scripts/profile-call-overhead.lisp、issue #190）'
    echo
    echo "各項目 ${NABLA_PROFILE_ITERATIONS:-200} 回"
    echo
    sed -n '/^## 測定条件/,$p' "${OUT}/profile.log"
    # 表では「未測定」としか出ないので、理由を残す
    grep -i ':status :unmeasured' "${OUT}/profile.sexp" | sed 's/^/    /' || true
  fi
} >> "${SUMMARY}"

echo "done" > "${OUT}/DONE"

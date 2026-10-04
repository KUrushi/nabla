#!/usr/bin/env bash
# Colab の GPU で local と cuda の数値一致を確かめる（issue #12）。
# 手元のマシン（Linux / macOS）で実行する。Colab CLI
# （https://github.com/googlecolab/google-colab-cli）を使う。
#
#   uv tool install google-colab-cli   # 初回のみ。初回の colab 実行で OAuth の認証がある
#   scripts/colab-gpu-check.sh                 # T4 で実行
#   scripts/colab-gpu-check.sh --gpu L4        # GPU を選ぶ（T4 / L4 / G4 / A100 / H100）
#   scripts/colab-gpu-check.sh --keep          # 終わっても VM を止めない（調べもの用）
#
# 流れ:
#   1. git archive HEAD でこのチェックアウトを固めて、新しい Colab セッションに送る
#   2. VM 上で scripts/colab/remote-gpu-check.sh をバックグラウンドで起動する
#      （IREE を --cuda でビルド → verify-iree.sh --cuda → large テスト
#       → 最大誤差の実測。初回は IREE の取得とビルドで 15〜30 分ほどかかる）
#   3. 終わるまで1分おきに進み具合を表示する
#   4. ログ一式を colab-gpu-out/<セッション名>/ に持ち帰り、VM を止める
#
# 結果の要約は colab-gpu-out/<セッション名>/summary.md。docs/iree-build.md の
# 「GPU で確かめる手順（issue #12）」の表を、ここに出た数字で埋める。
#
# 未コミットの変更は送られない（git archive HEAD を使うため）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU=T4
KEEP=0
POLL_SECONDS=60
MAX_WAIT_SECONDS=$((3 * 60 * 60))

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v colab >/dev/null 2>&1 || {
  echo "colab が見つからない。uv tool install google-colab-cli で入れる" >&2
  exit 2
}

SESSION="nabla-gpu-$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${REPO_ROOT}/colab-gpu-out/${SESSION}"
WORK="$(mktemp -d)"
mkdir -p "${OUT_DIR}"

log() { echo "[colab-gpu-check] $*"; }

# VM の上で Python を実行し、その標準出力を返す（colab exec は Python を実行する）
remote_py() {
  local timeout="$1"; shift
  printf '%s\n' "$*" | colab exec -s "${SESSION}" --timeout "${timeout}"
}

cleanup() {
  rm -rf "${WORK}"
  if [[ "${KEEP}" -eq 0 ]]; then
    log "セッション ${SESSION} を止める"
    colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  else
    log "--keep: セッション ${SESSION} は動いたまま。colab stop -s ${SESSION} で止める"
  fi
}

log "リポジトリを固める（HEAD = $(git -C "${REPO_ROOT}" rev-parse --short HEAD)）"
git -C "${REPO_ROOT}" archive --format=tar --prefix=nabla/ HEAD | tar -C "${WORK}" -xf -
git -C "${REPO_ROOT}" rev-parse HEAD > "${WORK}/nabla/.nabla-commit"
tar -C "${WORK}" -czf "${WORK}/nabla.tar.gz" nabla

log "Colab セッション ${SESSION} を作る（GPU: ${GPU}）"
colab new -s "${SESSION}" --gpu "${GPU}"
trap cleanup EXIT

log "リポジトリを送る"
colab upload -s "${SESSION}" "${WORK}/nabla.tar.gz" /content/nabla.tar.gz

log "VM 上で remote-gpu-check.sh を起動する"
remote_py 120 "
import subprocess
subprocess.run(['bash', '-c', 'rm -rf /content/nabla /content/nabla-out && tar -C /content -xzf /content/nabla.tar.gz'], check=True)
subprocess.Popen(['bash', '-c', 'bash /content/nabla/scripts/colab/remote-gpu-check.sh > /content/nabla-run.log 2>&1'],
                 start_new_session=True)
print('started')
"

log "終わるまで待つ（${POLL_SECONDS} 秒おき、最大 $((MAX_WAIT_SECONDS / 60)) 分）"
waited=0
while :; do
  status="$(remote_py 60 "
import os
print('DONE' if os.path.exists('/content/nabla-out/DONE') else 'RUNNING')
try:
    print(open('/content/nabla-out/progress.log').read().strip().splitlines()[-1])
except Exception:
    pass
" 2>/dev/null || echo "POLL-FAILED")"
  log "$(echo "${status}" | tail -n 1)"
  if echo "${status}" | grep -q '^DONE'; then
    break
  fi
  if (( waited >= MAX_WAIT_SECONDS )); then
    log "時間切れ。途中までのログを持ち帰る"
    break
  fi
  sleep "${POLL_SECONDS}"
  waited=$((waited + POLL_SECONDS))
done

log "ログを持ち帰る"
remote_py 120 "
import subprocess
subprocess.run(['bash', '-c', 'cp /content/nabla-run.log /content/nabla-out/ 2>/dev/null; tar -C /content -czf /content/nabla-out.tar.gz nabla-out'], check=True)
print('packed')
"
colab download -s "${SESSION}" /content/nabla-out.tar.gz "${OUT_DIR}/nabla-out.tar.gz"
tar -C "${OUT_DIR}" -xzf "${OUT_DIR}/nabla-out.tar.gz"

log "結果: ${OUT_DIR}/nabla-out/summary.md"
echo
cat "${OUT_DIR}/nabla-out/summary.md"

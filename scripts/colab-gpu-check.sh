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
  printf '%s\n' "$*" > "${WORK}/remote.py"
  # colab exec の --timeout は VM 側の実行時間だけを区切る。CLI と VM の接続が
  # 詰まると手元で返ってこないことがあるので、手元でも時間を区切る。
  # 出力はいったんファイルに受ける。打ち切った colab の子プロセスが標準出力の
  # パイプを持ったまま残ると、呼び出し側の $(...) がいつまでも終わらないため。
  local status=0
  with_timeout $((timeout + 60)) colab exec -s "${SESSION}" --timeout "${timeout}" \
    < "${WORK}/remote.py" > "${WORK}/remote.out" 2>&1 || status=$?
  cat "${WORK}/remote.out"
  return "${status}"
}

# with_timeout <秒> <コマンド...>: 時間内に終わらなければ止めて 124 を返す。
# macOS には timeout(1) が無いので bash だけで書く。見張りのサブシェルの
# 出力は捨てる（$(...) で包まれたときに、見張りがパイプを開いたままにしないため）。
with_timeout() {
  local secs="$1"; shift
  "$@" &
  local pid=$!
  ( sleep "${secs}"; pkill -TERM -P "${pid}" 2>/dev/null; kill -TERM "${pid}" 2>/dev/null ) \
    > /dev/null 2>&1 &
  local watcher=$!
  local status=0
  wait "${pid}" || status=$?
  kill "${watcher}" 2>/dev/null || true
  wait "${watcher}" 2>/dev/null || true
  if (( status == 143 )); then
    echo "[colab-gpu-check] ${secs} 秒で応答が無かったので打ち切った: $1 $2" >&2
    return 124
  fi
  return "${status}"
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

# VM に送るのはコミット済みの HEAD だけ。HEAD に必要なファイルが無いと、
# VM 側の処理が起動直後に失敗する（未コミットのファイルは送られない）。
for f in scripts/colab/remote-gpu-check.sh scripts/colab/measure-cross-device.lisp \
         tests/iree/cross-device-test.lisp; do
  if ! git -C "${REPO_ROOT}" cat-file -e "HEAD:${f}" 2>/dev/null; then
    echo "HEAD（$(git -C "${REPO_ROOT}" rev-parse --short HEAD)）に ${f} が無い。" >&2
    echo "このスクリプトを含むブランチ（claude/hopeful-fermi-11e9mw）をチェックアウトしてから実行する" >&2
    exit 2
  fi
done
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  log "注意: 未コミットの変更がある。VM には HEAD の内容だけを送る"
fi

log "リポジトリを固める（HEAD = $(git -C "${REPO_ROOT}" rev-parse --short HEAD)）"
# git archive の出力をそのまま送る。手元で展開して tar し直すと、macOS の
# tar が拡張属性を AppleDouble（._foo.lisp）として入れ、VM 上で
# tests/regressions/*.lisp のグロブに引っかかって load が落ちる。
COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
git -C "${REPO_ROOT}" archive --format=tar.gz --prefix=nabla/ -o "${WORK}/nabla.tar.gz" HEAD

log "Colab セッション ${SESSION} を作る（GPU: ${GPU}）"
colab new -s "${SESSION}" --gpu "${GPU}"
trap cleanup EXIT

log "リポジトリを送る"
colab upload -s "${SESSION}" "${WORK}/nabla.tar.gz" /content/nabla.tar.gz

log "VM 上で remote-gpu-check.sh を起動する"
remote_py 120 "
import subprocess
subprocess.run(['bash', '-c', 'rm -rf /content/nabla /content/nabla-out && tar -C /content -xzf /content/nabla.tar.gz'
                ' && find /content/nabla -name \"._*\" -delete'], check=True)
open('/content/nabla/.nabla-commit', 'w').write('${COMMIT}\\n')
p = subprocess.Popen(['bash', '-c', 'bash /content/nabla/scripts/colab/remote-gpu-check.sh > /content/nabla-run.log 2>&1'],
                     start_new_session=True)
open('/content/nabla-run.pid', 'w').write(str(p.pid))
print('started', p.pid)
"

log "終わるまで待つ（${POLL_SECONDS} 秒おき、最大 $((MAX_WAIT_SECONDS / 60)) 分）"
# 問い合わせの出力（colab の "[colab] ..." のメッセージは標準出力に出る）は
# すべて poll.log に残す。Colab 側で VM が消える
# （"Session not found"）と以後の問い合わせはすべて失敗するので、
# MAX_POLL_FAILURES 回続けて失敗したらセッションの有無を確かめ、消えて
# いれば待つのをやめる。
POLL_LOG="${OUT_DIR}/poll.log"
MAX_POLL_FAILURES=3
failures=0
waited=0
while :; do
  if status="$(remote_py 60 "
import os
def alive():
    try:
        os.kill(int(open('/content/nabla-run.pid').read()), 0)
        return True
    except Exception:
        return False
def last_line(path):
    try:
        return open(path, errors='replace').read().strip().splitlines()[-1]
    except Exception:
        return ''
if os.path.exists('/content/nabla-out/DONE'):
    print('DONE')
elif alive():
    print('RUNNING')
else:
    print('DIED')
    print('nabla-run.log: ' + last_line('/content/nabla-run.log'))
print(last_line('/content/nabla-out/progress.log') or '(まだ最初の手順に入っていない)')
" 2>&1)"; then
    failures=0
    { date -u +%FT%TZ; echo "${status}"; } >> "${POLL_LOG}"
    log "$(echo "${status}" | tail -n 1)"
    if echo "${status}" | grep -q '^DONE'; then
      break
    fi
    if echo "${status}" | grep -q '^DIED'; then
      log "VM 上の処理が DONE を書かずに終わった。ログを持ち帰る"
      echo "${status}" | grep '^nabla-run.log:' >&2 || true
      break
    fi
  else
    failures=$((failures + 1))
    { date -u +%FT%TZ; echo "POLL-FAILED"; echo "${status:-}"; } >> "${POLL_LOG}"
    log "問い合わせに失敗した（${failures} 回連続）: $(echo "${status:-}" | tail -n 1)"
    if (( failures >= MAX_POLL_FAILURES )); then
      if ! with_timeout 60 colab sessions 2>/dev/null | grep -q "${SESSION}"; then
        log "セッション ${SESSION} が Colab 側で消えている（VM が回収された）。"
        log "問い合わせの記録: ${POLL_LOG}。VM 側の記録: colab log -s ${SESSION} -n 50"
        exit 1
      fi
    fi
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
with_timeout 900 colab download -s "${SESSION}" /content/nabla-out.tar.gz "${OUT_DIR}/nabla-out.tar.gz"
tar -C "${OUT_DIR}" -xzf "${OUT_DIR}/nabla-out.tar.gz"

log "結果: ${OUT_DIR}/nabla-out/summary.md"
echo
cat "${OUT_DIR}/nabla-out/summary.md"

#!/usr/bin/env bash
# PJRT C API プラグイン（CPU、--cuda で CUDA も）を third_party/pjrt.lock で
# 固定した wheel から取得し、NABLA_PJRT_HOME に展開する。
#
# 必要なのは curl + unzip + sha256sum だけ（Python は要らない）。
#
#   scripts/fetch-pjrt.sh                 # CPU プラグインだけ
#   scripts/fetch-pjrt.sh --cuda          # CUDA プラグインも
#   scripts/fetch-pjrt.sh --keep-wheels   # ダウンロードした wheel を残す
#
# 環境変数:
#   NABLA_PJRT_HOME       インストール先。既定 ~/.local/share/nabla/pjrt-<cpu 版>
#                         配置: $NABLA_PJRT_HOME/cpu/xla_cpu_pjrt.so
#                               $NABLA_PJRT_HOME/cuda/xla_cuda_plugin.so
#   NABLA_PJRT_WHEEL_DIR  wheel の保存先。既定 ${XDG_CACHE_HOME:-~/.cache}/nabla/pjrt-wheel
#
# 冪等: 展開済みの .so があれば何もしない。wheel は sha256 が合えば再利用する。
# sha256 が lock と違えば失敗する（TLS 検証は無効化しない。プロキシの CA は
# curl の既定の設定に従う）。wheel は展開後に消す（--keep-wheels で残す）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="${REPO_ROOT}/third_party/pjrt.lock"

want_cuda=0
keep_wheels=0
for arg in "$@"; do
  case "$arg" in
    --cuda) want_cuda=1 ;;
    --keep-wheels) keep_wheels=1 ;;
    -h|--help) sed -n '2,/^set -/{/^set -/!p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "fetch-pjrt.sh: 不明な引数: $arg" >&2; exit 2 ;;
  esac
done

# lock の key=value を読む（source しない: 値をシェルとして評価させない）。
lock_get() {
  local value
  value="$(sed -n "s/^$1=//p" "$LOCK" | head -n1)"
  [ -n "$value" ] || { echo "fetch-pjrt.sh: $LOCK に $1 が無い" >&2; exit 1; }
  printf '%s\n' "$value"
}

PJRT_HOME="${NABLA_PJRT_HOME:-${HOME:?HOME が未設定です}/.local/share/nabla/pjrt-$(lock_get cpu_version)}"
WHEEL_DIR="${NABLA_PJRT_WHEEL_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/nabla/pjrt-wheel}"

fetch_one() {
  local kind="$1" dest_name="$2"
  local url sha member dest wheel actual tmp
  url="$(lock_get "${kind}_url")"
  sha="$(lock_get "${kind}_sha256")"
  member="$(lock_get "${kind}_member")"
  dest="${PJRT_HOME}/${kind}/${dest_name}"

  if [ -s "$dest" ]; then
    echo "fetch-pjrt.sh: ${kind}: 展開済み: $dest"
    return 0
  fi

  mkdir -p "$WHEEL_DIR" "${PJRT_HOME}/${kind}"
  wheel="${WHEEL_DIR}/$(basename "$url")"
  if [ -f "$wheel" ] && [ "$(sha256sum "$wheel" | cut -d' ' -f1)" = "$sha" ]; then
    echo "fetch-pjrt.sh: ${kind}: wheel は取得済み: $wheel"
  else
    echo "fetch-pjrt.sh: ${kind}: ダウンロード: $url"
    tmp="${wheel}.part"
    curl -fsSL --proto '=https' --retry 3 --retry-all-errors -o "$tmp" "$url"
    mv -f "$tmp" "$wheel"
  fi

  actual="$(sha256sum "$wheel" | cut -d' ' -f1)"
  if [ "$actual" != "$sha" ]; then
    echo "fetch-pjrt.sh: ${kind}: sha256 が一致しない: 期待 $sha 実際 $actual ($wheel)" >&2
    rm -f "$wheel"
    exit 1
  fi
  echo "fetch-pjrt.sh: ${kind}: sha256 OK"

  # 一時ファイルへ展開してから rename するので、途中で落ちても壊れた .so は残らない。
  tmp="${dest}.part"
  unzip -p "$wheel" "$member" > "$tmp"
  chmod 0755 "$tmp"
  mv -f "$tmp" "$dest"
  echo "fetch-pjrt.sh: ${kind}: 展開: $dest"
  if [ "$keep_wheels" -eq 0 ]; then rm -f "$wheel"; fi
}

fetch_one cpu xla_cpu_pjrt.so
if [ "$want_cuda" -eq 1 ]; then
  fetch_one cuda xla_cuda_plugin.so
fi
echo "fetch-pjrt.sh: NABLA_PJRT_HOME=${PJRT_HOME}"

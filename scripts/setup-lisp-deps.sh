#!/usr/bin/env bash
# 依存の準備（初回のみ）。
#
# - apt のパッケージは root で実行したときだけ apt-get install する。
#   root でなければ、必要なパッケージ名を表示するだけで続行する。
# - check-it と optima は GitHub になく apt にも無いので、固定コミットで
#   git clone する。$NABLA_LISP_DEPS の下に置き、既にあれば
#   fetch + checkout するだけにする（べき等）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${NABLA_LISP_DEPS:-${HOME:?NABLA_LISP_DEPS も HOME も未設定です}/.local/share/nabla/lisp-deps}"

APT_PACKAGES=(
  cl-fiveam
  cl-cffi
  cl-alexandria
  cl-trivial-garbage
  cl-lparallel
  cl-closer-mop
  cl-bordeaux-threads
  # nabla/iree のランタイムバインディングが使う cffi-libffi（cl-cffi に
  # 同梱）のビルドに必要。libffi-dev がないと cffi-libffi のロード時に
  # groveller が C コンパイルに失敗する。
  libffi-dev
)

# name url pin の3つ組。
GIT_DEPS=(
  "optima https://github.com/m2ym/optima 373b245b928c1a5cce91a6cb5bfe5dd77eb36195"
  "check-it https://github.com/DalekBaldwin/check-it b79c9103665be3976915b56b570038f03486e62f"
)

install_apt_packages() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "root ではないので apt install はスキップします。次のパッケージを別途インストールしてください:"
    printf '  %s\n' "${APT_PACKAGES[@]}"
    return
  fi
  local to_install=()
  for pkg in "${APT_PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
      to_install+=("$pkg")
    fi
  done
  if [ "${#to_install[@]}" -eq 0 ]; then
    echo "apt のパッケージはすべて既にインストールされています。"
    return
  fi
  apt-get update
  apt-get install -y "${to_install[@]}"
}

clone_git_dep() {
  local name="$1" url="$2" pin="$3"
  local dir="$DEPS_DIR/$name"
  if [ -d "$dir/.git" ]; then
    echo "$name: 既存のクローンを $pin に合わせます。"
    git -C "$dir" fetch --quiet origin "$pin" || git -C "$dir" fetch --quiet
    git -C "$dir" checkout --quiet "$pin"
  else
    echo "$name: $url を $pin に clone します。"
    mkdir -p "$DEPS_DIR"
    git clone --quiet "$url" "$dir"
    git -C "$dir" checkout --quiet "$pin"
  fi
}

install_apt_packages

for entry in "${GIT_DEPS[@]}"; do
  # shellcheck disable=SC2086
  clone_git_dep $entry
done

echo
echo "準備ができました。テストを実行するときは、次の CL_SOURCE_REGISTRY を使います:"
echo "  CL_SOURCE_REGISTRY=\"${REPO_ROOT}//:${DEPS_DIR}//:\""
echo "（scripts/run-tests.sh はこれを自動で設定します）"

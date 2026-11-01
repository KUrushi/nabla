#!/usr/bin/env bash
# 依存の準備（初回のみ）。
#
# - apt のパッケージは root で実行したときだけ apt-get install する。
#   root でなければ、必要なパッケージ名を表示するだけで続行する。
# - check-it・optima・png-read・cl-jpeg は apt に無いので、固定コミットで
#   git clone する。$NABLA_LISP_DEPS の下に置き、既にあれば
#   fetch + checkout するだけにする（べき等）。
# - cl-jpeg は safety 2 でコンパイルするよう直した fork（KUrushi/cl-jpeg の bounds-checked）を使う。
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
  # src/compile-cache.lisp（vmfb ディスクキャッシュ、issue #10）がキャッシュ
  # キーの SHA-256 に使う。
  cl-ironclad
  # nabla/data/image（issue #283）の PNG のデコーダ png-read の依存。
  cl-babel
  cl-chipz
  cl-iterate
)

# name url pin の3つ組。
# png-read と cl-jpeg は nabla/data/image（issue #283）の PNG / JPEG のデコーダ。
# どちらも apt に無い（ubuntu noble で確かめた）。
# cl-jpeg は上流（sharplispers/cl-jpeg）の cd92e94 に1コミット足した fork（issue #315）。
# 上流は jpeg.lisp の *optimize*（全関数の declare に #.*optimize* で配られる）で全体を
# (safety 0) でコンパイルし、入力を検証しない箇所（ハフマン表・成分の番号、符号の長さ、
# AC 係数の位置）があるので、壊れた JPEG で範囲外のメモリを読み書きし、Memory fault や
# ヒープの破壊でプロセスごと落ちる。fork は safety を 2 にする（1 は cl-jpeg の (speed 3) より
# 低いので SBCL が型の検査を弱め、SBCL 2.2.9 の x86-64 ではなお落ちた）。fork の master は
# 上流のまま保ち、直したコミットは bounds-checked ブランチに置く。
GIT_DEPS=(
  "optima https://github.com/m2ym/optima 373b245b928c1a5cce91a6cb5bfe5dd77eb36195"
  "check-it https://github.com/DalekBaldwin/check-it b79c9103665be3976915b56b570038f03486e62f"
  "png-read https://github.com/Ramarren/png-read ec29f38a689972b9f1373f13bbbcd6b05deada88"
  "cl-jpeg https://github.com/KUrushi/cl-jpeg 8617e260b5cf91dee1e17582c38af7d0958551e9"
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
    # 取得元を変えたとき（cl-jpeg を上流から fork に移したときなど）に、新しい取得元から取る。
    git -C "$dir" remote set-url origin "$url"
    git -C "$dir" fetch --quiet origin "$pin" || git -C "$dir" fetch --quiet
    # cl-jpeg を fork に移す前のクローンには、以前のこのスクリプト（patch_cl_jpeg_safety）が
    # safety を書き換えた jpeg.lisp が残っているので、戻してから移る（残っていると checkout が
    # 拒む）。ほかの変更は捨てない。同じコミットのままなら触らない（ファイルの mtime を
    # 変えず、ASDF に毎回コンパイルし直させない）。
    if [ "$name" = cl-jpeg ] &&
       [ "$(git -C "$dir" rev-parse HEAD)" != "$(git -C "$dir" rev-parse "$pin^{commit}")" ]; then
      git -C "$dir" checkout --quiet -- jpeg.lisp
    fi
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

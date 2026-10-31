#!/usr/bin/env bash
# 依存の準備（初回のみ）。
#
# - apt のパッケージは root で実行したときだけ apt-get install する。
#   root でなければ、必要なパッケージ名を表示するだけで続行する。
# - check-it・optima・png-read・cl-jpeg は apt に無いので、固定コミットで
#   git clone する。$NABLA_LISP_DEPS の下に置き、既にあれば
#   fetch + checkout するだけにする（べき等）。
# - cl-jpeg は checkout した後で safety を 2 に書き換える（patch_cl_jpeg_safety）。
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
GIT_DEPS=(
  "optima https://github.com/m2ym/optima 373b245b928c1a5cce91a6cb5bfe5dd77eb36195"
  "check-it https://github.com/DalekBaldwin/check-it b79c9103665be3976915b56b570038f03486e62f"
  "png-read https://github.com/Ramarren/png-read ec29f38a689972b9f1373f13bbbcd6b05deada88"
  "cl-jpeg https://github.com/sharplispers/cl-jpeg cd92e94ff45218c3833143aaebe7522f19079cef"
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
    # cl-jpeg の固定コミットが変わったときは、patch_cl_jpeg_safety が書き換えた jpeg.lisp を
    # 戻してから移る（書き換えが残っていると checkout が拒む）。ほかの変更は捨てない。
    # 同じコミットのままなら触らない（書き換えたファイルの mtime を変えず、ASDF に毎回
    # コンパイルし直させない）。
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

# cl-jpeg は jpeg.lisp の *optimize*（全関数の declare に #.*optimize* で配られる）で
# 全体を (safety 0) でコンパイルするので、壊れた JPEG で範囲外のメモリを読み書きし、
# コンディションではなく Memory fault でプロセスごと落ちる（issue #283）。checkout の
# 直後に safety を 2 に書き換える。1 では足りない: cl-jpeg の (speed 3) より低いので
# SBCL は型の検査を弱め、SBCL 2.2.9 の x86-64 では safety 1 でもなお落ちた。
# 書き換えたことは nabla/data/image が JPEG をデコードする前に確かめる
# （src/data/image/image.lisp の %check-cl-jpeg-safety）。本来は fork で直すもので、
# これは fork ができるまでの暫定。外部の patch ファイルにしないのは、CI のキャッシュの
# キー（このスクリプトのハッシュ）に書き換えの中身を入れるため。
# 2 回目以降は書き換え済みの行を見て何もしない（べき等）。上流の行が変わって
# 書き換える行が見つからなければ、エラーで止める。
patch_cl_jpeg_safety() {
  local file="$DEPS_DIR/cl-jpeg/jpeg.lisp"
  local original="(defparameter *optimize* '(optimize (safety 0) (space 0) (debug 0) (speed 3))))"
  local patched="(defparameter *optimize* '(optimize (safety 2) (space 0) (debug 0) (speed 3))))"
  if [ "$(grep -cF -- "$patched" "$file")" -eq 1 ] && ! grep -qF -- "$original" "$file"; then
    echo "cl-jpeg: *optimize* は既に safety 2 です。"
    return
  fi
  if [ "$(grep -cF -- "$original" "$file")" -ne 1 ]; then
    echo "cl-jpeg: $file に次の行がちょうど1つ見つからないので、safety を書き換えられません:" >&2
    echo "  $original" >&2
    echo "上流の行が変わったときは、patch_cl_jpeg_safety の original / patched を直してください。" >&2
    exit 1
  fi
  # sed -i は GNU と BSD で書き方が違うので、awk の固定文字列の置換で一時ファイルに書く。
  awk -v from="$original" -v to="$patched" '{
    i = index($0, from)
    if (i > 0) $0 = substr($0, 1, i - 1) to substr($0, i + length(from))
    print
  }' "$file" > "$file.tmp"
  mv "$file.tmp" "$file"
  if [ "$(grep -cF -- "$patched" "$file")" -ne 1 ] || grep -qF -- "$original" "$file"; then
    echo "cl-jpeg: $file の safety の書き換えに失敗しました。" >&2
    exit 1
  fi
  echo "cl-jpeg: *optimize* の safety を 0 から 2 に書き換えました。"
}

install_apt_packages

for entry in "${GIT_DEPS[@]}"; do
  # shellcheck disable=SC2086
  clone_git_dep $entry
done
patch_cl_jpeg_safety

echo
echo "準備ができました。テストを実行するときは、次の CL_SOURCE_REGISTRY を使います:"
echo "  CL_SOURCE_REGISTRY=\"${REPO_ROOT}//:${DEPS_DIR}//:\""
echo "（scripts/run-tests.sh はこれを自動で設定します）"

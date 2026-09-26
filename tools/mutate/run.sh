#!/usr/bin/env bash
# tools/mutate/run.sh -- mutation testing runner の CLI
#
# 使い方:
#   tools/mutate/run.sh [--system NAME] [--base REF] [--trials N] [--timeout SEC]
#                        [--test-system SYSTEM] [--test-form FORM]
#                        [FILE[:START-END]...]
#
# 引数を何も渡さなければ、--base（既定 main）から HEAD までの git diff で
# 変わった .lisp の行が対象になる。FILE[:START-END] を渡すと、そのファイル
# （の指定した行範囲）だけが対象になる。
#
# --system は asdf:test-system を呼ぶときの対象システム名（既定 nabla）。
# --test-system を渡すと、その ASDF システムを runner の前に読み込む。
# 渡さなければ既定で "${SYSTEM}/tests"（例: nabla/tests）を読み込む。
# --test-form を渡すと、それを eval した結果（関数）をテスト実行関数として使う。
# 渡さなければ NABLA.TESTS.SUPPORT:RUN-TESTS を実行時に探す
# （nabla-mutate.asd の DEFAULT-TEST-FUNCTION を見よ）。
#
# 終了コード: mutation score が 0.8 以上（変異させられる定義が1つもない
# ときは 1 として扱われる）なら 0、そうでなければ 1。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# git diff が出すパスや exclusions.lisp などの相対パスの解決を、呼び出し時の
# カレントディレクトリに左右されないようにする（サブディレクトリから
# 実行してもリポジトリ直下から実行したのと同じ結果にする）。
cd "$REPO"
NABLA_LISP_DEPS_DIR="${NABLA_LISP_DEPS:-$HOME/.local/share/nabla/lisp-deps}"
# tools/mutate/sample//: を明示的に足しているのは、SBCL の ASDF が
# 深い階層の .asd を再帰探索で見つけないことがあるため
# （tools/mutate/nabla-mutate.asd は見つかるが、その下の
# tools/mutate/sample/nabla-mutate-sample.asd までは見つからない環境があった）。
export CL_SOURCE_REGISTRY="${REPO}//:${REPO}/tools/mutate/sample//:${NABLA_LISP_DEPS_DIR}//:"

SYSTEM="nabla"
BASE_REF="main"
TRIALS="20"
TIMEOUT="300"
TEST_SYSTEM=""
TEST_FORM=""
declare -a RANGE_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --system) SYSTEM="$2"; shift 2 ;;
    --base) BASE_REF="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --test-system) TEST_SYSTEM="$2"; shift 2 ;;
    --test-form) TEST_FORM="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) RANGE_ARGS+=("$1"); shift ;;
  esac
done
RANGE_ARGS+=("$@")

# FILE を Lisp の文字列リテラルの中身として安全に埋め込めるようにする
# （\ と " をエスケープする）。パスにこの2文字を含む環境は稀だが、
# 埋め込まずに壊れた eval フォームを作ってしまうよりはよい。
lisp_escape_string() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  printf '%s' "$s"
}

if [ -z "$TEST_SYSTEM" ]; then
  TEST_SYSTEM="${SYSTEM}/tests"
fi

# FILE[:START-END] を ("path" start end) の Lisp リストへ変換する。
ranges_lisp="nil"
if [ "${#RANGE_ARGS[@]}" -gt 0 ]; then
  entries=""
  for arg in "${RANGE_ARGS[@]}"; do
    file="${arg%%:*}"
    if [ "$file" = "$arg" ]; then
      start=1
      end=1000000000
    else
      range="${arg#*:}"
      start="${range%%-*}"
      end="${range#*-}"
    fi
    entries="${entries} (list \"$(lisp_escape_string "$file")\" ${start} ${end})"
  done
  ranges_lisp="(list${entries})"
fi

load_forms="(asdf:load-system \"nabla-mutate\")"
if [ -n "$TEST_SYSTEM" ]; then
  load_forms="${load_forms} (asdf:load-system \"$(lisp_escape_string "$TEST_SYSTEM")\")"
fi

if [ -n "$TEST_FORM" ]; then
  test_function_form="${TEST_FORM}"
else
  test_function_form="#'nabla.mutate:default-test-function"
fi

run_form="(let ((report (nabla.mutate:run
                            :system \"$(lisp_escape_string "$SYSTEM")\"
                            :test-function ${test_function_form}
                            :ranges ${ranges_lisp}
                            :base-ref \"$(lisp_escape_string "$BASE_REF")\"
                            :timeout-seconds ${TIMEOUT}
                            :trials ${TRIALS})))
             (uiop:quit (if (>= (nabla.mutate:mutation-score report) 4/5) 0 1)))"

exec sbcl --non-interactive \
  --eval "(require :asdf)" \
  --eval "(progn ${load_forms})" \
  --eval "${run_form}"

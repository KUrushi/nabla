#!/usr/bin/env bash
# nabla のテストを実行する。既定は small + medium。
#
#   scripts/run-tests.sh
#   NABLA_TEST_SIZES=large scripts/run-tests.sh
#
# ql:quickload は使わない。CL_SOURCE_REGISTRY でリポジトリと
# $NABLA_LISP_DEPS を ASDF に見せ、asdf:load-system で読み込む。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${NABLA_LISP_DEPS:-${HOME:?NABLA_LISP_DEPS も HOME も未設定です}/.local/share/nabla/lisp-deps}"

# REPO_ROOT は非再帰 (末尾 /) にする。再帰 (//) にすると、.claude/worktrees/
# 以下にチェックアウトされた別のワークツリーの nabla.asd までここから
# 見えてしまい、メインのチェックアウトから実行したときに、意図しない
# ワークツリーの nabla.asd をロードすることがある。DEPS_DIR（fiveam や
# cffi の apt レジストリより下にある check-it / optima）は複数階層
# ネストしているので、そちらだけ再帰 (//) にする。末尾の //: は apt の
# 既定レジストリ（fiveam / cffi など）も残す。
export CL_SOURCE_REGISTRY="${REPO_ROOT}/:${DEPS_DIR}//:"

CHECK_ASD_SOURCE_FILE="(let ((expected (merge-pathnames \"nabla.asd\" #p\"${REPO_ROOT}/\"))
                (actual (asdf:system-source-file \"nabla\")))
            (unless (equal (truename expected) (truename actual))
              (error \"scripts/run-tests.sh: nabla.asd が想定と違う場所から見つかった。期待: ~A 実際: ~A\" expected actual)))"

main_status=0
sbcl --non-interactive \
  --eval '(require :asdf)' \
  --eval "${CHECK_ASD_SOURCE_FILE}" \
  --eval '(asdf:load-system "nabla/tests")' \
  --eval '(asdf:load-system "nabla/iree/tests")' \
  --eval '(uiop:quit (if (nabla.tests.support:run-tests) 0 1))' \
  || main_status=$?

# tests/iree/jit-test.lisp（:NABLA.ISOLATED-MEDIUM スイート）は、他の
# medium テストと同じ SBCL プロセスで実行すると in-process の
# libIREECompiler.so が壊れて落ちることが分かっている（issue #68）ので、
# NABLA_TEST_SIZES に "medium" が含まれるときだけ、上とは別の SBCL
# プロセスでこのスイートを実行する。"medium" が含まれるかどうかの判定は
# ここで NABLA_TEST_SIZES を独自に parse し直さず、常に別プロセスを
# 起動した上で NABLA.TESTS.SUPPORT:SIZES-FROM-ENV（既定値・区切り文字の
# 扱いの正本）自身に判定させる（判定ロジックの二重管理を避ける）。medium が
# 含まれなければそのプロセスは何もせず 0 で終了する。
isolated_status=0
sbcl --non-interactive \
  --eval '(require :asdf)' \
  --eval "${CHECK_ASD_SOURCE_FILE}" \
  --eval '(asdf:load-system "nabla/iree/tests")' \
  --eval '(uiop:quit (if (member :medium (nabla.tests.support:sizes-from-env))
                         (if (fiveam:run! :nabla.isolated-medium) 0 1)
                         0))' \
  || isolated_status=$?

if [ "${main_status}" -ne 0 ] || [ "${isolated_status}" -ne 0 ]; then
  exit 1
fi
exit 0

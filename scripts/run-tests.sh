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

exec sbcl --non-interactive \
  --eval '(require :asdf)' \
  --eval "(let ((expected (merge-pathnames \"nabla.asd\" #p\"${REPO_ROOT}/\"))
                (actual (asdf:system-source-file \"nabla\")))
            (unless (equal (truename expected) (truename actual))
              (error \"scripts/run-tests.sh: nabla.asd が想定と違う場所から見つかった。期待: ~A 実際: ~A\" expected actual)))" \
  --eval '(asdf:load-system "nabla/tests")' \
  --eval '(asdf:load-system "nabla/iree/tests")' \
  --eval '(uiop:quit (if (nabla.tests.support:run-tests) 0 1))'

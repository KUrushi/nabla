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

# 末尾の //: は apt の既定レジストリ（fiveam / cffi など）も残す。
export CL_SOURCE_REGISTRY="${REPO_ROOT}//:${DEPS_DIR}//:"

exec sbcl --non-interactive \
  --eval '(require :asdf)' \
  --eval '(asdf:load-system "nabla/tests")' \
  --eval '(asdf:load-system "nabla/iree/tests")' \
  --eval '(uiop:quit (if (nabla.tests.support:run-tests) 0 1))'

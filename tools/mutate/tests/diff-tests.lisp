;;;; diff-tests.lisp -- ranges-from-git-diff の対象範囲フィルタ

(in-package #:nabla.mutate.tests)

(in-suite :nabla-mutate)

(test in-mutation-scope-p-keeps-only-src-lisp-excluding-iree-and-pjrt
  "mutation testing の対象範囲は src/ 以下の .lisp のうち、
src/iree/ と src/pjrt/ の CFFI バインディングを除いたもの。"
  (is-true (nabla.mutate::%in-mutation-scope-p "src/core/primitives/add.lisp"))
  (is-true (nabla.mutate::%in-mutation-scope-p "src/foo.lisp"))
  (is-false (nabla.mutate::%in-mutation-scope-p "src/iree/bindings.lisp"))
  (is-false (nabla.mutate::%in-mutation-scope-p "src/pjrt/bindings.lisp"))
  (is-false (nabla.mutate::%in-mutation-scope-p "tests/core/add-tests.lisp"))
  (is-false (nabla.mutate::%in-mutation-scope-p "tools/mutate/src/runner.lisp"))
  ;; "src/iree-utils.lisp" のような似た名前は誤って除外しない
  ;; (src/iree/ という「/」区切りのディレクトリ一致だけを見る)
  (is-true (nabla.mutate::%in-mutation-scope-p "src/iree-utils.lisp")))

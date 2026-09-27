;;;; package.lisp -- nabla-mutate/tests のパッケージ定義とスイート

(defpackage #:nabla.mutate.tests
  (:use #:cl #:fiveam #:check-it)
  (:shadowing-import-from #:check-it #:*num-trials*)
  (:export #:run-tests))

(in-package #:nabla.mutate.tests)

(def-suite :nabla-mutate)

(defun run-tests ()
  "nabla-mutate/tests のすべてのテストを実行し、すべて通れば T、
1つでも落ちれば NIL を返す。"
  (let ((results (run :nabla-mutate)))
    (explain! results)
    (fiveam:results-status results)))

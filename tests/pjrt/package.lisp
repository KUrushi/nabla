;;;; nabla/pjrt/tests のパッケージ。

(defpackage #:nabla.pjrt.tests
  (:use #:cl #:fiveam #:check-it #:nabla.tests.support #:nabla.pjrt)
  (:shadowing-import-from #:check-it #:*num-trials*)
  (:documentation
   "nabla/pjrt のテストを置くパッケージ。"))

(in-package #:nabla.pjrt.tests)

;;; nabla/tests の tests/regressions.lisp はこのパッケージより先に読まれるので、自分の回帰ケースはここで読む
(load-package-regressions '#:nabla.pjrt.tests)

(in-suite :nabla.medium)

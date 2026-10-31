;;;; nabla/nn/tests のパッケージ。

(defpackage #:nabla.nn.tests
  (:use #:cl #:fiveam #:check-it #:nabla.tests.support #:nabla.nn)
  (:shadowing-import-from #:check-it #:*num-trials*)
  ;; check-it も CHAIN を export している。ここで使うのは nabla.nn のオプティマイザの chain
  (:shadowing-import-from #:nabla.nn #:chain)
  (:documentation
   "nabla/nn のテストを置くパッケージ。"))

(in-package #:nabla.nn.tests)

;;; nabla/tests の tests/regressions.lisp はこのパッケージより先に読まれるので、自分の回帰ケースはここで読む
(load-package-regressions '#:nabla.nn.tests)

(in-suite :nabla.small)

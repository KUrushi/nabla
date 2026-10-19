;;;; nabla/nn/tests のパッケージ。

(defpackage #:nabla.nn.tests
  (:use #:cl #:fiveam #:check-it #:nabla.tests.support #:nabla.nn)
  (:shadowing-import-from #:check-it #:*num-trials*)
  ;; check-it も CHAIN を export している。ここで使うのは nabla.nn のオプティマイザの chain
  (:shadowing-import-from #:nabla.nn #:chain)
  (:documentation
   "nabla/nn のテストを置くパッケージ。"))

(in-package #:nabla.nn.tests)

(in-suite :nabla.small)

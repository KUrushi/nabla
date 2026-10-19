;;;; nabla/nn/tests のパッケージ。

(defpackage #:nabla.nn.tests
  (:use #:cl #:fiveam #:check-it #:nabla.tests.support #:nabla.nn)
  (:shadowing-import-from #:check-it #:*num-trials*)
  (:documentation
   "nabla/nn のテストを置くパッケージ。"))

(in-package #:nabla.nn.tests)

(in-suite :nabla.small)

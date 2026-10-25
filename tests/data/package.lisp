;;;; nabla/data/tests のパッケージ（issue #279）。

(defpackage #:nabla.data.tests
  (:use #:cl #:fiveam #:check-it #:nabla.tests.support #:nabla.data)
  (:shadowing-import-from #:check-it #:*num-trials*)
  (:documentation
   "nabla/data のテストを置くパッケージ。"))

(in-package #:nabla.data.tests)

(in-suite :nabla.small)

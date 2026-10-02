;;;; nabla/pjrt/tests のパッケージ。

(defpackage #:nabla.pjrt.tests
  (:use #:cl #:fiveam #:check-it #:nabla.tests.support #:nabla.pjrt)
  (:shadowing-import-from #:check-it #:*num-trials*)
  (:documentation
   "nabla/pjrt のテストを置くパッケージ。"))

(in-package #:nabla.pjrt.tests)

(in-suite :nabla.medium)

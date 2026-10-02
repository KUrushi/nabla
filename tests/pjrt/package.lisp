;;;; nabla/pjrt/tests のパッケージ。

(defpackage #:nabla.pjrt.tests
  (:use #:cl #:fiveam #:nabla.tests.support #:nabla.pjrt)
  (:documentation
   "nabla/pjrt のテストを置くパッケージ。"))

(in-package #:nabla.pjrt.tests)

(in-suite :nabla.medium)

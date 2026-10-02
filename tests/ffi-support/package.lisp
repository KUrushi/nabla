;;;; nabla/ffi-support/tests のパッケージ。nabla/iree は :use しない
;;;; （nabla/iree をロードせずに通ることが完了条件、issue #79）。

(defpackage #:nabla.ffi-support.tests
  (:use #:cl #:fiveam #:nabla.tests.support)
  (:documentation "nabla/ffi-support のテストを置くパッケージ。"))

(in-package #:nabla.ffi-support.tests)


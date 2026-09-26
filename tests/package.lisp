;;;; nabla/tests のパッケージ。

(defpackage #:nabla.tests
  (:use #:cl #:fiveam #:check-it #:nabla.tests.support)
  (:shadowing-import-from #:check-it #:*num-trials*)
  (:documentation
   "nabla コアと tests/support のテストを置くパッケージ。"))

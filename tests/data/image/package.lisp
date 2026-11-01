;;;; nabla/data/image/tests のパッケージ（issue #283）。

(defpackage #:nabla.data.image.tests
  (:use #:cl #:fiveam #:check-it #:nabla.tests.support #:nabla.data #:nabla.data.image)
  (:shadowing-import-from #:check-it #:*num-trials*)
  (:documentation
   "nabla/data/image のテストを置くパッケージ。"))

(in-package #:nabla.data.image.tests)

;;; nabla/tests の tests/regressions.lisp はこのパッケージより先に読まれるので、自分の回帰ケースはここで読む
(load-package-regressions '#:nabla.data.image.tests)

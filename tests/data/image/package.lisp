;;;; nabla/data/image/tests のパッケージ（issue #283）。

(defpackage #:nabla.data.image.tests
  (:use #:cl #:fiveam #:check-it #:nabla.tests.support #:nabla.data #:nabla.data.image)
  (:shadowing-import-from #:check-it #:*num-trials*)
  (:documentation
   "nabla/data/image のテストを置くパッケージ。"))

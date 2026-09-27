;;;; nabla/iree/tests のパッケージ（プレースホルダー）。
;;;;
;;;; wave 2 で IREE のバインディングを実装するときに、このスイートへ
;;;; テストを追加していく。medium スイートに参加する準備だけしておく。

(defpackage #:nabla.iree.tests
  (:use #:cl #:fiveam #:check-it #:nabla.tests.support #:nabla.iree)
  (:shadowing-import-from #:check-it #:*num-trials*)
  (:documentation
   "nabla/iree のテストを置くパッケージ。"))

(in-package #:nabla.iree.tests)

(in-suite :nabla.medium)

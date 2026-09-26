;;;; nabla/test-support: 各テストシステムが共有するテストの土台。
;;;;
;;;; FiveAM のスイート定義、check-it の生成器、乱数配列の生成、
;;;; 許容誤差つきの比較、回帰テストのファイル管理をここに置く。
;;;; nabla/iree/tests のような後続のテストシステムは、この
;;;; nabla/test-support に依存し、nabla/tests には依存しない。

(defpackage #:nabla.tests.support
  (:use #:cl #:fiveam #:check-it)
  ;; fiveam と check-it はどちらも *num-trials* を export している。
  ;; check-it 側を優先する（PBT の試行回数はこちらで制御するため）。
  (:shadowing-import-from #:check-it #:*num-trials*)
  (:export
   ;; スイート実行
   #:run-tests
   #:sizes-from-env
   ;; dtype と許容誤差
   #:*dtypes*
   #:dtype-tolerance
   ;; array-spec 生成器
   #:array-spec
   #:make-array-spec
   #:array-spec-shape
   #:array-spec-dtype
   #:array-spec-rank
   #:make-random-array
   #:decode-element
   ;; 許容誤差つきの比較
   #:allclose
   #:approx=
   ;; 回帰テスト
   #:regression-path))

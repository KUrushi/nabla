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
   ;; check-it::*size* にクランプされない一様な整数・実数の生成器
   ;; （.claude/skills/nabla-testing/references/properties.md の
   ;; 「check-it の (integer lo hi) / (real lo hi) の落とし穴」参照）
   #:uniform-integer
   #:uniform-real
   #:make-uniform-integer-generator
   #:make-uniform-real-generator
   ;; array-spec 生成器
   #:array-spec
   #:make-array-spec
   #:array-spec-shape
   #:array-spec-dtype
   #:array-spec-rank
   #:make-random-array
   #:decode-element
   #:decode-array
   ;; 許容誤差つきの比較
   #:allclose
   #:approx=
   ;; 参照実装（テストの中でループを書かずに済むオラクル）
   #:reference-add
   #:reference-matmul
   #:reference-reduce-sum
   ;; issue #31 p4: reshape / broadcast-in-dim / transpose
   #:reference-reshape
   #:reference-broadcast-in-dim
   #:reference-transpose
   ;; フェイク backend（issue #9。nabla:backend プロトコルの参照実装）
   #:fake-backend
   #:fake-backend-compile-count
   #:fake-array
   ;; 一時ディレクトリ（issue #10 のディスクキャッシュのテストなどで使う）
   #:with-temporary-directory
   ;; 回帰テスト
   #:regression-path))

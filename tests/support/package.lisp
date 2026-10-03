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
   ;; 子 SBCL プロセスの環境（child-sbcl.lisp）
   #:%child-source-registry
   #:*child-sbcl-forwarded-env-vars*
   #:%forward-env-vars
   ;; スイート実行
   #:run-tests
   #:sizes-from-env
   ;; dtype と許容誤差
   #:*dtypes*
   #:dtype-tolerance
   #:graph-worst-float-dtype
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
   ;; issue #31 p1
   #:reference-sub
   #:reference-mul
   #:reference-div
   ;; issue #31 p2
   #:reference-neg
   #:reference-exp
   #:reference-log
   #:reference-tanh
   #:reference-max
   #:reference-min
   ;; issue #31 p3
   #:reference-compare
   #:reference-select
   #:reference-matmul
   #:reference-reduce-sum
   ;; issue #31 p4: reshape / broadcast-in-dim / transpose
   #:reference-reshape
   #:reference-broadcast-in-dim
   #:reference-transpose
   ;; issue #31 p5: dot-general
   #:reference-dot-general
   ;; issue #31 p6: reduce-sum / reduce-max
   #:reference-reduce-max
   ;; フェイク backend（issue #9。nabla:backend プロトコルの参照実装）
   #:fake-backend
   #:fake-backend-compile-count
   #:fake-array
   ;; 一時ディレクトリ（issue #10 のディスクキャッシュのテストなどで使う）
   #:with-temporary-directory
   ;; 回帰テスト
   #:regression-path
   ;; StableHLO emitter の medium PBT が使う、実プリミティブ上のレシピ生成器
   ;; （issue #33、wave 3 s1）
   #:primitive-graph-recipe
   #:build-primitive-graph
   #:primitive-recipe-eqn-count
   #:replay-recipe-avals
   ;; レシピが使う dtype（束縛すると f64 だけの graph も作れる。issue #76）
   #:*primitive-recipe-dtypes*
   ;; 自動微分のテスト支援（issue #76）
   #:central-difference-jvp
   #:central-difference-gradient
   #:*central-difference-step*
   #:*autodiff-rtol*
   #:*autodiff-atol*
   #:inner-product
   #:random-tangent
   #:random-cotangent
   #:scalar-loss-function
   #:scalar-loss-oracle
   ;; テスト専用の高階プリミティブ（issue #127）
   #:test-call-subgraph
   #:test-while-capture))

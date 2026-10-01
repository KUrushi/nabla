;;;; -*- Mode: LISP -*-
;;;;
;;;; nabla: Common Lisp × IREE の深層学習ライブラリ。
;;;;
;;;; フェーズ0（IREE 疎通）の骨格。システムは nabla / nabla/test-support /
;;;; nabla/tests / nabla/iree / nabla/iree/tests の5つ。nabla/pjrt / nabla/nn /
;;;; nabla/data はまだ作らない。
;;;;
;;;; 1つの defsystem に1つの :components エントリを1行、で揃えている。
;;;; 後続のフェーズでファイルを足すときは、この形を崩さない。

(defsystem "nabla"
  :description "Common Lisp で書く JAX 相当の深層学習ライブラリ（コア。実行系の実装は知らない）"
  :author "KUrushi"
  :license "Apache-2.0"
  :depends-on ("ironclad" "trivial-garbage" (:require "sb-cltl2"))
  :components ((:file "src/package")
               (:file "src/dtype")
               ;; bf16 / f16 のビット列 <-> single-float 変換（issue #38、u4）
               (:file "src/float16")
               (:file "src/aval")
               ;; IR と defprimitive（issue #29、u1a）
               (:file "src/primitive")
               (:file "src/ir")
               ;; graph の印字と読み込み（issue #29、u1b）
               (:file "src/ir-print")
               ;; 二項算術プリミティブ add / sub / mul / div（issue #31 p1）
               (:file "src/primitives/common")
               (:file "src/primitives/arith")
               ;; issue #31 p4: reshape / broadcast-in-dim / transpose
               (:file "src/primitives/shape-common")
               (:file "src/primitives/shape")
               ;; 単項プリミティブ neg / exp / log / tanh（issue #31 p2）
               (:file "src/primitives/unary")
               ;; compare / select / convert（issue #31 p3）
               (:file "src/primitives/compare")
               ;; stop-gradient（issue #80）
               (:file "src/primitives/stop-gradient")
               ;; issue #31 p5: dot-general
               (:file "src/primitives/dot")
               ;; graph の eager 評価（issue #39、e0。プリミティブ（p1..p6）が
               ;; まだ無いので src/ir-print の直後に置く。将来のプリミティブは
               ;; この行より前に足す）
               (:file "src/eval")
               ;; issue #31 p6: reduce-sum / reduce-max
               (:file "src/primitives/reduce")
               ;; トレーサ（issue #32、t1）
               (:file "src/trace")
               (:file "src/trace-ops")
               (:file "src/walk")
               ;; if を select に、配列レベルの公開 API（issue #32、t2）
               (:file "src/array-api")
               ;; 自動微分の骨格: symbolic zero とルール種別（issue #77、77a）
               (:file "src/ad/zero")
               (:file "src/ad/rules")
               ;; graph のインライン化と不要 eqn の削除（issue #77、77b）
               (:file "src/ad/inline")
               ;; jvp 変換と、要素ごとのプリミティブの jvp ルール（issue #77、77c）
               (:file "src/ad/jvp")
               (:file "src/ad/rules-elementwise")
               ;; linearize / transpose / vjp（issue #82）
               (:file "src/ad/linearize")
               (:file "src/ad/transpose")
               ;; 形状・縮約・dot-general の jvp ルール（issue #81）
               (:file "src/ad/rules-shape")
               ;; StableHLO テキスト emitter（issue #33、wave 3 s1）
               (:file "src/stablehlo")
               (:file "src/backend")
               (:file "src/compile-cache")
               ;; jit とインメモリのコンパイルキャッシュ（issue #34、wave 4 j1）
               (:file "src/jit"))
  :in-order-to ((test-op (test-op "nabla/tests"))))

(defsystem "nabla/test-support"
  :description "nabla の全テストシステムが共有するテストの土台（FiveAM のスイート、check-it の生成器、比較関数）"
  :depends-on ("nabla" "fiveam" "check-it")
  :components ((:file "tests/support/package")
               (:file "tests/support/suites")
               (:file "tests/support/uniform-generator")
               (:file "tests/support/dtypes")
               (:file "tests/support/array-spec")
               (:file "tests/support/random-array")
               (:file "tests/support/allclose")
               (:file "tests/support/reference")
               (:file "tests/support/fake-backend")
               (:file "tests/support/temporary-directory")
               (:file "tests/support/regression")
               ;; StableHLO emitter の medium PBT が使うレシピ生成器（issue #33、s1）
               (:file "tests/support/primitive-recipes")
               (:file "tests/support/autodiff")
               (:file "tests/support/run-tests")))

(defsystem "nabla/tests"
  :description "nabla コアの small/medium/large テスト"
  :depends-on ("nabla" "nabla/test-support")
  :components ((:file "tests/package")
               (:file "tests/support-test")
               (:file "tests/autodiff-support-test")
               (:file "tests/dtype-test")
               ;; bf16 / f16 のビット列 <-> single-float 変換（issue #38、u4）
               (:file "tests/float16-test")
               (:file "tests/aval-test")
               (:file "tests/backend-test")
               (:file "tests/compile-cache-test")
               ;; IR と defprimitive（issue #29、u1a）
               (:file "tests/test-primitives")
               (:file "tests/graph-recipes")
               (:file "tests/primitive-test")
               (:file "tests/ir-test")
               ;; graph の印字と読み込み（issue #29、u1b）
               (:file "tests/ir-print-test")
               ;; 二項算術プリミティブ add / sub / mul / div（issue #31 p1）
               (:file "tests/primitives/arith-test")
               ;; issue #31 p4: reshape / broadcast-in-dim / transpose
               (:file "tests/primitives/shape-test")
               ;; 単項プリミティブ neg / exp / log / tanh、max / min（issue #31 p2）
               (:file "tests/primitives/unary-test")
               ;; compare / select / convert（issue #31 p3）
               (:file "tests/primitives/compare-test")
               ;; issue #31 p5: dot-general
               (:file "tests/primitives/dot-test")
               ;; graph の eager 評価（issue #39、e0）
               (:file "tests/eval-test")
               ;; issue #31 p6: reduce-sum / reduce-max
               (:file "tests/primitives/reduce-test")
               (:file "tests/primitives/registry-test")
               ;; トレーサ（issue #32、t1）
               (:file "tests/walk-test")
               (:file "tests/trace-test")
               ;; if を select に、配列レベルの公開 API（issue #32、t2）
               (:file "tests/trace-if-test")
               (:file "tests/array-api-test")
               ;; graph のインライン化と不要 eqn の削除（issue #77、77b）
               (:file "tests/ad/inline-test")
               ;; StableHLO テキスト emitter（issue #33、wave 3 s1）
               (:file "tests/stablehlo-test")
               ;; 自動微分の骨格: symbolic zero とルール種別（issue #77、77a）
               (:file "tests/ad/zero-test")
               (:file "tests/ad/rules-test")
               ;; jvp 変換（issue #77、77c）
               (:file "tests/ad/jvp-test")
               ;; linearize / transpose / vjp（issue #82）
               (:file "tests/ad/linearize-test")
               (:file "tests/ad/transpose-test")

               ;; 要素演算の jvp ルール（issue #80）
               (:file "tests/ad/jvp-elementwise-test")
               (:file "tests/ad/stop-gradient-test")
               ;; 形状・縮約・dot-general の jvp ルール（issue #81）
               (:file "tests/ad/jvp-shape-test")
               ;; jit とインメモリのコンパイルキャッシュ（issue #34、wave 4 j1）
               (:file "tests/jit-test")
               ;; jit キャッシュのモジュール解放・並行性・defjit の :static-args（issue #71）
               (:file "tests/jit-cache-test")
               (:file "tests/regressions"))
  :perform (test-op (op c)
             (declare (ignore op c))
             (unless (funcall (intern "RUN-TESTS" :nabla.tests.support))
               (error "nabla/tests: 既定のテストスイートが失敗した"))))

(defsystem "nabla/iree"
  :description "IREE 連携（コンパイラとランタイムの埋め込み C API のバインディング）"
  :depends-on ("nabla" "cffi" "cffi-libffi" "trivial-garbage")
  :components ((:file "src/iree/package")
               ;; 浮動小数点例外トラップの一括マスク（issue #53、x1）
               (:file "src/iree/float-traps")
               (:file "src/iree/conditions")
               (:file "src/iree/compiler-ffi")
               (:file "src/iree/signals")
               (:file "src/iree/library")
               (:file "src/iree/compiler")
               (:file "src/iree/runtime-ffi")
               (:file "src/iree/status")
               (:file "src/iree/runtime")
               (:file "src/iree/device-array")
               (:file "src/iree/execute")
               (:file "src/iree/backend")))

;; nabla/iree/tests は nabla/tests から独立したシステム（システム構成は
;; 契約 §2 のとおり）。そのため (asdf:test-system "nabla") はこのシステムを
;; ロードしない。scripts/run-tests.sh は両方を明示的に load-system してから
;; run-tests を呼んでいるので、コマンドとして使う分には問題ない。
(defsystem "nabla/iree/tests"
  :description "nabla/iree のテスト"
  :depends-on ("nabla/iree" "nabla/test-support")
  :components ((:file "tests/iree/package")
               (:file "tests/iree/support")
               (:file "tests/iree/support-test")
               ;; finalizer-test はここ（compiler-test / runtime-test /
               ;; device-array-test / execute-test より前）に置く。どれも Lisp の
               ;; 関数としては、より前にロードされる src/iree/*.lisp にしか
               ;; 依存しないので、テストファイルの順序を変えても機能的な依存関係は
               ;; 壊れない。この順にした経緯は finalizer-test.lisp のトップの
               ;; コメントを参照: かつて発生していた fatal error（issue #5、LLVM が
               ;; SIGUSR2 のハンドラを上書きすることが原因）の発生頻度を下げる
               ;; 緩和策として置いたが、根本原因自体は src/iree/signals.lisp で
               ;; 修正済みなので、この順序はもう必須ではない。害も無いので変えて
               ;; いない。
               (:file "tests/iree/finalizer-test")
               (:file "tests/iree/backend-test")
               (:file "tests/iree/compile-cache-test")
               (:file "tests/iree/cross-device-test")
               (:file "tests/iree/compiler-test")
               (:file "tests/iree/runtime-test")
               (:file "tests/iree/runtime-cuda-test")
               (:file "tests/iree/device-array-test")
               (:file "tests/iree/execute-test")
               (:file "tests/iree/example-test")
               ;; StableHLO op 対応表（issue #30、u2）
               (:file "tests/iree/ops-test")
               ;; 1つの op だけを持つモジュールのビルダー（全プリミティブの medium
               ;; テストが共有する。issue #31 p1、#74 で p4 の複製を統合）
               (:file "tests/iree/primitive-support")
               ;; 二項算術プリミティブ add / sub / mul / div（issue #31 p1）
               (:file "tests/iree/arith-test")
               ;; issue #31 p4: reshape / broadcast-in-dim / transpose
               (:file "tests/iree/shape-test")
               ;; 単項プリミティブ neg / exp / log / tanh、max / min（issue #31 p2）
               (:file "tests/iree/unary-test")
               ;; compare / select / convert（issue #31 p3）
               (:file "tests/iree/compare-test")
               ;; issue #31 p5: dot-general
               (:file "tests/iree/dot-test")
               ;; issue #31 p6: reduce-sum / reduce-max
               (:file "tests/iree/reduce-test")
               ;; 浮動小数点例外トラップの一括マスク（issue #53、x1）
               (:file "tests/iree/float-traps-test")
               ;; StableHLO テキスト emitter（issue #33、wave 3 s1）
               (:file "tests/iree/stablehlo-test")
               ;; jvp 変換した graph の IREE 実行（issue #77、77c）
               (:file "tests/iree/jvp-test")
               ;; vjp 変換した graph の IREE 実行（issue #82）
               (:file "tests/iree/vjp-test")

               ;; 要素演算の jvp ルールの IREE 実行（issue #80）
               (:file "tests/iree/jvp-elementwise-test")
               (:file "tests/iree/jvp-shape-test")
               ;; defjit、compile-error のリスタート、end-to-end jit テスト（issue #34、wave 4 j2）
               (:file "tests/iree/jit-test")
               ;; jit キャッシュのモジュール解放・並行性・入れ子（issue #71）
               (:file "tests/iree/jit-cache-test")
               ;; f64 と :i1 の to-device / to-host / invoke を jit で通す（issue #72）
               (:file "tests/iree/jit-dtype-test")))

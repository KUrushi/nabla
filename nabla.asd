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
  :depends-on ("ironclad")
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
               ;; graph の eager 評価（issue #39、e0。プリミティブ（p1..p6）が
               ;; まだ無いので src/ir-print の直後に置く。将来のプリミティブは
               ;; この行より前に足す）
               (:file "src/eval")
               (:file "src/backend")
               (:file "src/compile-cache"))
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
               (:file "tests/support/run-tests")))

(defsystem "nabla/tests"
  :description "nabla コアの small/medium/large テスト"
  :depends-on ("nabla" "nabla/test-support")
  :components ((:file "tests/package")
               (:file "tests/support-test")
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
               ;; graph の eager 評価（issue #39、e0）
               (:file "tests/eval-test")
               (:file "tests/regressions"))
  :perform (test-op (op c)
             (declare (ignore op c))
             (unless (funcall (intern "RUN-TESTS" :nabla.tests.support))
               (error "nabla/tests: 既定のテストスイートが失敗した"))))

(defsystem "nabla/iree"
  :description "IREE 連携（コンパイラとランタイムの埋め込み C API のバインディング）"
  :depends-on ("nabla" "cffi" "cffi-libffi" "trivial-garbage")
  :components ((:file "src/iree/package")
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
               ;; 二項算術プリミティブ add / sub / mul / div（issue #31 p1）
               (:file "tests/iree/primitive-support")
               (:file "tests/iree/arith-test")
               ;; issue #31 p4: reshape / broadcast-in-dim / transpose
               (:file "tests/iree/shape-primitive-support")
               (:file "tests/iree/shape-test")
               ;; 単項プリミティブ neg / exp / log / tanh、max / min（issue #31 p2）
               (:file "tests/iree/unary-test")))

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
  :description "Common Lisp から IREE を叩く JAX 相当の深層学習ライブラリ（コア）"
  :author "KUrushi"
  :license "MIT"
  :depends-on ()
  :components ((:file "src/package")
               (:file "src/dtype")
               (:file "src/aval"))
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
               (:file "tests/support/regression")
               (:file "tests/support/run-tests")))

(defsystem "nabla/tests"
  :description "nabla コアの small/medium/large テスト"
  :depends-on ("nabla" "nabla/test-support")
  :components ((:file "tests/package")
               (:file "tests/support-test")
               (:file "tests/dtype-test")
               (:file "tests/aval-test")
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
               (:file "src/iree/execute")))

;; nabla/iree/tests は nabla/tests から独立したシステム（システム構成は
;; 契約 §2 のとおり）。そのため (asdf:test-system "nabla") はこのシステムを
;; ロードしない。scripts/run-tests.sh は両方を明示的に load-system してから
;; run-tests を呼んでいるので、コマンドとして使う分には問題ない。
(defsystem "nabla/iree/tests"
  :description "nabla/iree のテスト"
  :depends-on ("nabla/iree" "nabla/test-support")
  :components ((:file "tests/iree/package")
               (:file "tests/iree/support")
               (:file "tests/iree/compiler-test")
               (:file "tests/iree/runtime-test")
               (:file "tests/iree/runtime-cuda-test")
               (:file "tests/iree/device-array-test")
               (:file "tests/iree/execute-test")))

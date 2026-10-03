;;;; -*- Mode: LISP -*-
;;;;
;;;; nabla: Common Lisp × IREE の深層学習ライブラリ。
;;;;
;;;; システムは nabla / nabla/test-support / nabla/tests / nabla/ffi-support /
;;;; nabla/ffi-support/tests / nabla/iree / nabla/iree/tests / nabla/pjrt /
;;;; nabla/pjrt/tests（と nabla/nn / nabla/data の予定）。nabla/ffi-support は IREE と PJRT が
;;;; 共有する FFI 保護（issue #79）で、nabla/iree 無しでロードできる。
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
               (:file "src/jit")
               ;; grad / value-and-grad（issue #86。jitted-function を受けるので jit の後）
               (:file "src/ad/grad")

               ;; フェーズ3 anchor: issue #127（サブグラフを持つ eqn）。この issue のコンポーネントはこの下に足す



               ;; フェーズ3 anchor: issue #126（整数 dtype）。この issue のコンポーネントはこの下に足す



               ;; フェーズ3 anchor: issue #125（vmap の骨格）。この issue のコンポーネントはこの下に足す
               (:file "src/vmap")
               ;; バッチ化ルール（要素演算は #128、形状演算は #129）
               (:file "src/ad/rules-batch-elementwise")
               (:file "src/ad/rules-batch-shape")



               ;; フェーズ3 anchor: issue #128（要素演算のバッチ化ルール）。この issue のコンポーネントはこの下に足す



               ;; フェーズ3 anchor: issue #129（形状演算・縮約・dot-general のバッチ化ルール）。この issue のコンポーネントはこの下に足す



               ;; フェーズ3 anchor: issue #130（cond プリミティブ）。この issue のコンポーネントはこの下に足す
               (:file "src/primitives/cond")
               (:file "src/cond")



               ;; フェーズ3 anchor: issue #131（while-loop プリミティブ）。この issue のコンポーネントはこの下に足す
               (:file "src/while-loop")



               ;; フェーズ3 anchor: issue #132（scan プリミティブ）。この issue のコンポーネントはこの下に足す



               ;; フェーズ3 anchor: issue #133（rng-bit-generator）。この issue のコンポーネントはこの下に足す
               (:file "src/primitives/rng")



               ;; フェーズ3 anchor: issue #134（cond / while-loop の jvp）。この issue のコンポーネントはこの下に足す



               ;; フェーズ3 anchor: issue #135（scan の jvp）。この issue のコンポーネントはこの下に足す



               ;; フェーズ3 anchor: issue #136（PRNG の公開 API）。この issue のコンポーネントはこの下に足す
               (:file "src/primitives/bits")
               (:file "src/ad/rules-batch-rng")
               (:file "src/prng")



               ;; フェーズ3 anchor: issue #137（dotimes / loop を scan に展開）。この issue のコンポーネントはこの下に足す



               ;; フェーズ3 anchor: issue #138（per-example 勾配）。この issue のコンポーネントはこの下に足す



               ;; フェーズ3 anchor: issue #139（scan の linearize と transpose）。この issue のコンポーネントはこの下に足す



               ;; フェーズ3 anchor: issue #140（制御構造のバッチ化ルール）。この issue のコンポーネントはこの下に足す



               ;; フェーズ3 anchor: issue #141（RNN の e2e）。この issue のコンポーネントはこの下に足す
               )
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
               (:file "tests/support/child-sbcl")
               (:file "tests/support/regression")
               ;; StableHLO emitter の medium PBT が使うレシピ生成器（issue #33、s1）
               (:file "tests/support/primitive-recipes")
               (:file "tests/support/autodiff")
               ;; テスト専用の高階プリミティブ（issue #127）
               (:file "tests/support/subgraph-primitive")
               ;; vmap の参照実装（issue #125）
               (:file "tests/support/vmap")
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
               (:file "tests/load-warnings-test")
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
               ;; ベンチマークスクリプトの出力の読み書き（issue #89）
               (:file "tests/bench-test")
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
               ;; 線形プリミティブの transpose ルール（issue #83）
               (:file "tests/ad/transpose-rules-test")
               ;; grad / value-and-grad（issue #86）
               (:file "tests/ad/grad-test")
               ;; jit とインメモリのコンパイルキャッシュ（issue #34、wave 4 j1）
               (:file "tests/jit-test")
               ;; jit キャッシュのモジュール解放・並行性・defjit の :static-args（issue #71）
               (:file "tests/jit-cache-test")
               ;; フェーズ3 anchor: issue #127（サブグラフを持つ eqn）。この issue のテストはこの下に足す
               (:file "tests/subgraph-test")



               ;; フェーズ3 anchor: issue #126（整数 dtype）。この issue のテストはこの下に足す
               (:file "tests/integer-dtype-test")



               ;; フェーズ3 anchor: issue #125（vmap の骨格）。この issue のテストはこの下に足す
               (:file "tests/vmap-test")



               ;; フェーズ3 anchor: issue #128（要素演算のバッチ化ルール）。この issue のテストはこの下に足す
               (:file "tests/vmap-elementwise-test")



               ;; フェーズ3 anchor: issue #129（形状演算・縮約・dot-general のバッチ化ルール）。この issue のテストはこの下に足す
               (:file "tests/vmap-shape-test")



               ;; フェーズ3 anchor: issue #130（cond プリミティブ）。この issue のテストはこの下に足す
               (:file "tests/cond-test")



               ;; フェーズ3 anchor: issue #131（while-loop プリミティブ）。この issue のテストはこの下に足す
               (:file "tests/while-loop-test")



               ;; フェーズ3 anchor: issue #132（scan プリミティブ）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #133（rng-bit-generator）。この issue のテストはこの下に足す
               (:file "tests/primitives/rng-test")



               ;; フェーズ3 anchor: issue #134（cond / while-loop の jvp）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #135（scan の jvp）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #136（PRNG の公開 API）。この issue のテストはこの下に足す
               (:file "tests/primitives/bits-test")
               (:file "tests/prng-test")



               ;; フェーズ3 anchor: issue #137（dotimes / loop を scan に展開）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #138（per-example 勾配）。この issue のテストはこの下に足す
               (:file "tests/per-example-test")



               ;; フェーズ3 anchor: issue #139（scan の linearize と transpose）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #140（制御構造のバッチ化ルール）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #141（RNN の e2e）。この issue のテストはこの下に足す

               (:file "tests/regressions"))
  :perform (test-op (op c)
             (declare (ignore op c))
             (unless (funcall (intern "RUN-TESTS" :nabla.tests.support))
               (error "nabla/tests: 既定のテストスイートが失敗した"))))

(defsystem "nabla/iree"
  :description "IREE 連携（コンパイラとランタイムの埋め込み C API のバインディング）"
  :depends-on ("nabla" "nabla/ffi-support" "cffi" "cffi-libffi" "trivial-garbage")
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

(defsystem "nabla/ffi-support"
  :description "C ライブラリ呼び出しの共有の保護（シグナルハンドラと浮動小数点トラップ。IREE / 将来の PJRT が使う）"
  :depends-on ("cffi")
  :components ((:file "src/ffi-support/package")
               (:file "src/ffi-support/float-traps")
               (:file "src/ffi-support/signals")))

;; nabla/ffi-support/tests は nabla/iree をロードせずに通る（issue #79）。
(defsystem "nabla/ffi-support/tests"
  :description "nabla/ffi-support のテスト（nabla/iree 無しで動くことも確かめる）"
  :depends-on ("nabla/ffi-support" "nabla/test-support")
  :components ((:file "tests/ffi-support/package")
               (:file "tests/ffi-support/ffi-support-test")))

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
               ;; 線形プリミティブの transpose ルール（vjp）の IREE 実行（issue #83）
               (:file "tests/iree/vjp-rules-test")
               ;; defjit、compile-error のリスタート、end-to-end jit テスト（issue #34、wave 4 j2）
               (:file "tests/iree/jit-test")
               ;; jit キャッシュのモジュール解放・並行性・入れ子（issue #71）
               (:file "tests/iree/jit-cache-test")
               ;; f64 と :i1 の to-device / to-host / invoke を jit で通す（issue #72）
               (:file "tests/iree/jit-dtype-test")
               ;; (jit (grad f)) の IREE 実行（issue #86）
               (:file "tests/iree/grad-test")
               ;; 2層 MLP の学習 end-to-end（issue #88）
               (:file "tests/iree/mlp-train-test")
               ;; ベンチマークスクリプトを小さな設定で1回走らせる（issue #89）
               (:file "tests/iree/bench-test")

               ;; フェーズ3 anchor: issue #127（サブグラフを持つ eqn）。この issue のテストはこの下に足す
               (:file "tests/iree/subgraph-test")



               ;; フェーズ3 anchor: issue #126（整数 dtype）。この issue のテストはこの下に足す
               (:file "tests/iree/integer-test")



               ;; フェーズ3 anchor: issue #125（vmap の骨格）。この issue のテストはこの下に足す
               (:file "tests/iree/vmap-test")



               ;; フェーズ3 anchor: issue #128（要素演算のバッチ化ルール）。この issue のテストはこの下に足す
               (:file "tests/iree/vmap-elementwise-test")



               ;; フェーズ3 anchor: issue #129（形状演算・縮約・dot-general のバッチ化ルール）。この issue のテストはこの下に足す
               (:file "tests/iree/vmap-shape-test")



               ;; フェーズ3 anchor: issue #130（cond プリミティブ）。この issue のテストはこの下に足す
               (:file "tests/iree/cond-test")



               ;; フェーズ3 anchor: issue #131（while-loop プリミティブ）。この issue のテストはこの下に足す
               (:file "tests/iree/while-loop-test")



               ;; フェーズ3 anchor: issue #132（scan プリミティブ）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #133（rng-bit-generator）。この issue のテストはこの下に足す
               (:file "tests/iree/rng-test")



               ;; フェーズ3 anchor: issue #134（cond / while-loop の jvp）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #135（scan の jvp）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #136（PRNG の公開 API）。この issue のテストはこの下に足す
               (:file "tests/iree/prng-test")



               ;; フェーズ3 anchor: issue #137（dotimes / loop を scan に展開）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #138（per-example 勾配）。この issue のテストはこの下に足す
               (:file "tests/iree/per-example-test")



               ;; フェーズ3 anchor: issue #139（scan の linearize と transpose）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #140（制御構造のバッチ化ルール）。この issue のテストはこの下に足す



               ;; フェーズ3 anchor: issue #141（RNN の e2e）。この issue のテストはこの下に足す
               ))

(defsystem "nabla/pjrt"
  :description "PJRT 連携（プラグインの .so を dlopen し、クライアント・デバイス・device-array を扱い、StableHLO のコンパイル・ロード・実行を行う）"
  :depends-on ("nabla" "nabla/ffi-support" "cffi" "trivial-garbage" "ironclad")
  :components ((:file "src/pjrt/package")
               (:file "src/pjrt/library")
               (:file "src/pjrt/ffi")
               (:file "src/pjrt/client")
               (:file "src/pjrt/device-array")
               (:file "src/pjrt/executable")
               (:file "src/pjrt/backend")))

;; nabla/pjrt/tests も nabla/iree/tests と同じく nabla/tests から独立している。
(defsystem "nabla/pjrt/tests"
  :description "nabla/pjrt のテスト（プラグインが無ければスキップ。NABLA_REQUIRE_PJRT が空でなければ失敗）"
  :depends-on ("nabla/pjrt" "nabla/test-support" (:require "sb-posix"))
  :components ((:file "tests/pjrt/package")
               (:file "tests/pjrt/support")
               (:file "tests/pjrt/support-test")
               ;; CPU プラグインの dlopen と PJRT API の版（issue #78）
               (:file "tests/pjrt/smoke-test")
               ;; FFI 定義とヘッダの照合、クライアント・backend・device-array（issue #85）
               (:file "tests/pjrt/ffi-test")
               (:file "tests/pjrt/backend-test")
               ;; コンパイル・ロード・実行、jit、fingerprint（issue #87）
               (:file "tests/pjrt/executable-test")
               ;; ベンチマークスクリプトを小さな設定で1回走らせる（issue #89）
               (:file "tests/pjrt/bench-test")
               (:file "tests/pjrt/integer-test")
               ;; フェーズ3 anchor: issue #133（rng-bit-generator）
               (:file "tests/pjrt/rng-test")
               ;; フェーズ3 anchor: issue #136（PRNG の公開 API）
               (:file "tests/pjrt/prng-test")))

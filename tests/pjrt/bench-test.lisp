;;;; scripts/bench-backends.sh を小さな設定で1回走らせ、出力が期待する形（env 1件と、
;;;; 各 metric の result）であることを確かめる（issue #89）。数値そのものは主張しない。
;;;; IREE 側は tests/iree/bench-test.lisp（NABLA_REQUIRE_IREE の有無に依らず PJRT だけで動くようにするため分けてある）。

(in-package #:nabla.pjrt.tests)

(defun %run-bench (backend)
  "scripts/bench-backends.sh を BACKEND だけ・small・5ステップ・1回で走らせ、
(values 終了コード 標準出力 標準エラー) を返す。"
  (multiple-value-bind (out err code)
      (uiop:run-program (list "timeout" "-k" "5" "600"
                              (namestring (asdf:system-relative-pathname "nabla" "scripts/bench-backends.sh"))
                              "--backends" backend "--configs" "small" "--steps" "5" "--warmup" "2" "--reps" "1")
                        :output :string :error-output :string :ignore-error-status t)
    (values code out err)))

(define-pjrt-test bench/script-runs-on-pjrt-cpu
    "bench-backends.sh --backends pjrt-cpu が成功し、出力の表に測定条件と、初期化・段ごとのコンパイル・ステップの項目が全部ある。"
  (skip-unless-pjrt)
  (multiple-value-bind (code out err) (%run-bench "pjrt-cpu")
    (fiveam:is (zerop code) "終了コード ~D: ~A" code err)
    (dolist (needle '("## 測定条件" "pjrt-cpu" "pjrt-plugin-sha256" "init/plugin-load" "init/client-create"
                      "init/plugin-sha256" "stage/trace" "stage/emit-stablehlo" "stage/backend-compile"
                      "stage/backend-load" "jit/first-call" "jit/second-call" "step/full" "step/jitted-call"))
      (fiveam:is (search needle out) "出力に ~S が無い:~%~A" needle out))))

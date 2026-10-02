;;;; scripts/bench-backends.sh を iree-local で小さな設定で1回走らせ、出力が期待する形であることを
;;;; 確かめる（issue #89）。数値そのものは主張しない。

(in-package #:nabla.iree.tests)

(define-iree-test bench/script-runs-on-iree-local
    "bench-backends.sh --backends iree-local が成功し、出力の表に測定条件と、初期化・段ごとのコンパイル・ステップの項目が全部ある。"
  (skip-unless-iree :library :both)
  (multiple-value-bind (out err code)
      (uiop:run-program (list "timeout" "-k" "5" "600"
                              (namestring (asdf:system-relative-pathname "nabla" "scripts/bench-backends.sh"))
                              "--backends" "iree-local" "--configs" "small" "--steps" "5" "--warmup" "2" "--reps" "1")
                        :output :string :error-output :string :ignore-error-status t)
    (fiveam:is (zerop code) "終了コード ~D: ~A" code err)
    (dolist (needle '("## 測定条件" "iree-local" "iree-compiler-revision" "init/compiler-load" "init/device-create"
                      "stage/trace" "stage/emit-stablehlo" "stage/backend-compile" "stage/backend-load"
                      "jit/first-call" "jit/second-call" "step/full" "step/jitted-call"))
      (fiveam:is (search needle out) "出力に ~S が無い:~%~A" needle out))))

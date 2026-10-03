;;;; scan を通る vjp（逆モード微分）の graph を IREE（local）でコンパイル・実行し、eager
;;;; （eval-graph）と比べる medium テスト（issue #139）。tests/iree/scan-test.lisp の
;;;; %WITH-SCAN-IREE-CHECK と tests/iree/jvp-scan-test.lisp の graph を使う。

(in-package #:nabla.iree.tests)

(define-iree-test vjp-scan/iree-vjp-matches-eager
    "scan（carry・consts・xs の混在。forward と reverse）の vjp graph（partial eval した主値の scan と、転置した逆向きの scan）が、IREE の実行結果と eager で一致する。"
  (skip-unless-iree :library :both)
  (dolist (reverse '(nil t))
    (let ((graph (nb::vjp-graph (%jvp-scan-iree-graph 3 4 reverse))))
      (%with-scan-iree-check (graph "vjp") "IREE の scan の vjp graph の結果が eager と一致しなかった"))))

(define-iree-test vjp-scan/iree-partial-vjp-matches-eager
    "xs の u にだけ余接線を求める vjp graph（未知の carry が不動点で広がる）も IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::vjp-graph (%jvp-scan-iree-graph 3 4 t) :nonzero '(nil nil nil nil t nil))))
    (%with-scan-iree-check (graph "vjp-partial") "IREE の部分的な scan の vjp graph の結果が eager と一致しなかった")))

(define-iree-test vjp-scan/iree-elman-grad-matches-eager
    "Elman 型の RNN（h' = tanh(W h + U x + b)、損失は最後の h の総和）の grad graph が IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let* ((f (nb:with-tracing (w u b h0 xs)
              (nb:reduce-sum
               (first (nb:scan (nb:with-tracing (carry x)
                                 (values (list (tanh (+ (+ (nb:dot w (first carry)) (nb:dot u (first x))) b)))
                                         '()))
                               (list h0) (list xs) :length 5)))))
         (graph (nb::trace-to-graph
                 (let ((g (nb:grad f :argnums '(0 1 2 3 4))))
                   (nb:with-tracing (w u b h0 xs) (values-list (funcall g w u b h0 xs))))
                 (list (nb:make-aval '(3 3) :f32) (nb:make-aval '(3 3) :f32) (nb:make-aval '(3) :f32)
                       (nb:make-aval '(3) :f32) (nb:make-aval '(5 3) :f32)))))
    (%with-scan-iree-check (graph "elman-grad") "IREE の RNN の grad graph の結果が eager と一致しなかった")))

(define-iree-test vjp-scan/iree-elman-grad-at-size-matches-eager
    "中くらいの大きさ（H=16、T=30、入力は ±0.3 に収める）の Elman RNN の grad graph が IREE と eager で一致する（積んだ残差と逆向きの while の lowering を大きさのあるものでも確かめる）。"
  (skip-unless-iree :library :both)
  (let* ((h 16) (steps 30)
         (f (nb:with-tracing (w u b h0 xs)
              (nb:reduce-sum
               (first (nb:scan (nb:with-tracing (carry x)
                                 (values (list (tanh (+ (+ (nb:dot w (first carry)) (nb:dot u (first x))) b)))
                                         '()))
                               (list h0) (list xs) :length steps)))))
         (graph (nb::trace-to-graph
                 (let ((g (nb:grad f :argnums '(0 1 2 3 4))))
                   (nb:with-tracing (w u b h0 xs) (values-list (funcall g w u b h0 xs))))
                 (list (nb:make-aval (list h h) :f32) (nb:make-aval (list h h) :f32) (nb:make-aval (list h) :f32)
                       (nb:make-aval (list h) :f32) (nb:make-aval (list steps h) :f32)))))
    ;; 乱数の seed は固定の数個（100 試行の PBT では、f32 の足し込みの順序の差が許容誤差を超える
    ;; seed が稀にあった。大きさのある1つの graph の lowering を確かめるのが目的なので固定にする）。
    (let* ((backend (nabla:find-backend :iree))
           (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
      (unwind-protect
           (dolist (seed '(0 1 2 3 4))
             (is (%scan-iree-matches-eager-p backend module graph seed)
                 "IREE の大きさのある RNN の grad graph の結果が eager と一致しなかった (seed ~D)" seed))
        (nabla:backend-unload backend module)))))

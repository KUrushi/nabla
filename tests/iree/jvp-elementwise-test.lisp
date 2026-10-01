;;;; 要素演算の jvp ルールを使った jvp graph の IREE 経由 end-to-end テスト
;;;; （issue #80）。
;;;;
;;;; sub mul div exp log tanh max min select stop-gradient（と合成）の jvp-graph を
;;;; emit-stablehlo → IREE でコンパイル・実行した全出力（主値 ++ 接線）が、
;;;; 同じ graph の eval-graph と一致する。入力は正の値（log / div の定義域）。

(in-package #:nabla.iree.tests)

(defparameter *jvp-iree-elementwise-functions*
  (list (cons :sub (nb:with-tracing (x y) (- x y)))
        (cons :mul (nb:with-tracing (x y) (* x y)))
        (cons :div (nb:with-tracing (x y) (/ x y)))
        (cons :exp (nb:with-tracing (x y) (+ (exp x) y)))
        (cons :log (nb:with-tracing (x y) (+ (log x) y)))
        (cons :tanh (nb:with-tracing (x y) (+ (tanh x) y)))
        (cons :max (nb:with-tracing (x y) (max x y)))
        (cons :min (nb:with-tracing (x y) (min x y)))
        (cons :select (nb:with-tracing (x y) (nb:where (< x y) (* x y) (exp y))))
        (cons :composite (nb:with-tracing (x y) (tanh (* x (exp y)))))
        ;; stop-gradient（optimization_barrier）。:I1 の値も通す。
        (cons :stop-gradient (nb:with-tracing (x y) (+ (* (nb:stop-gradient x) y) (* x y))))
        (cons :stop-gradient-i1 (nb:with-tracing (x y) (nb:where (nb:stop-gradient (< x y)) x y))))
  "名前と with-tracing した2入力関数の対。")

(define-iree-test jvp/iree-elementwise-rules-match-eval-graph
    "各要素演算の jvp-graph（全接線と x だけの接線）を IREE で実行した全出力が
eval-graph と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 3))
    (loop for (name . fn) in *jvp-iree-elementwise-functions*
          do (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                           (lambda (seed)
                             (let* ((shape (loop for k from 1 to (mod seed 3) collect (1+ (mod (+ seed k) 3))))
                                    (aval (nb:make-aval shape :f32))
                                    (spec (make-array-spec shape :f32))
                                    (graph (nb:trace-to-graph fn (list aval aval)))
                                    (arrays (loop for i below 4
                                                  collect (make-random-array spec :seed (+ seed i) :domain :positive))))
                               (and (%jvp-iree-matches-eval-graph-p backend (nb::jvp-graph graph) arrays)
                                    (%jvp-iree-matches-eval-graph-p backend (nb::jvp-graph graph :nonzero '(t nil))
                                                                    (subseq arrays 0 3)))))
                           :regression-id jvp/iree-elementwise-rules-match-eval-graph
                           :regression-file (regression-path "iree-jvp-elementwise" :package "NABLA.IREE.TESTS"))
                 "~S" name))))

;;;; jvp 変換した graph の IREE 経由 end-to-end テスト（issue #77、77c）。
;;;;
;;;; nb::jvp-graph が作った graph（主値 ++ 接線の多出力）を emit-stablehlo →
;;;; backend-compile/load/invoke した結果が、同じ graph の eval-graph と
;;;; 一致する。ルールを持つ実プリミティブは今は add / neg だけ（#80 で増える）。
;;;; shape ごとに1回の IREE コンパイル（約350ms）になるので試行回数は小さく抑える。

(in-package #:nabla.iree.tests)

(defun %jvp-iree-matches-eval-graph-p (backend graph arrays)
  "GRAPH（f32）を ARRAYS で IREE 実行した全出力が eval-graph の結果と一致するか。"
  (let* ((module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph))))
         (device-arrays '())
         (results '()))
    (unwind-protect
         (progn
           (setf device-arrays (mapcar (lambda (a) (to-device a backend :dtype :f32)) arrays))
           (setf results (multiple-value-list (apply #'nabla:backend-invoke backend module "main" device-arrays)))
           (let ((expected (multiple-value-list (apply #'nb:eval-graph graph arrays))))
             (and (= (length results) (length expected))
                  (every (lambda (r e) (allclose (to-host r) e :dtype :f32)) results expected))))
      (dolist (r results) (release-device-array r))
      (dolist (da device-arrays) (release-device-array da))
      (nabla:backend-unload backend module))))

(define-iree-test jvp/iree-matches-eval-graph
    "(lambda (x y) (- (+ x y))) を trace-to-graph → jvp-graph（全接線と、x だけの
接線の2通り）→ emit-stablehlo → IREE で実行した結果は、同じ graph の
eval-graph と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 8))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((shape (loop repeat (mod seed 3) for k from 1 collect (1+ (mod (+ seed k) 3))))
                           (aval (nb:make-aval shape :f32))
                           (spec (make-array-spec shape :f32))
                           (graph (nb:trace-to-graph (nb:with-tracing (x y) (- (+ x y))) (list aval aval)))
                           (arrays (loop for i below 4 collect (make-random-array spec :seed (+ seed i)))))
                      (and (%jvp-iree-matches-eval-graph-p backend (nb::jvp-graph graph) arrays)
                           (%jvp-iree-matches-eval-graph-p backend (nb::jvp-graph graph :nonzero '(t nil))
                                                           (subseq arrays 0 3)))))
                  :regression-id jvp/iree-matches-eval-graph
                  :regression-file (regression-path "iree-jvp-matches-eval-graph" :package "NABLA.IREE.TESTS")))))

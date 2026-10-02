;;;; 実プリミティブの transpose ルールを通した vjp の IREE 経由 end-to-end テスト
;;;; （issue #83）。
;;;;
;;;; sub / mul / div（片側が既知）/ select / broadcast-in-dim / reshape / transpose /
;;;; reduce-sum と、非線形の tanh / exp / compare を1つの関数に通し、vjp-graph の
;;;; 結果が emit-stablehlo → IREE で eval-graph と一致する。
;;;; %vjp-iree-matches-eval-graph-p は vjp-test.lisp のもの。

(in-package #:nabla.iree.tests)

(define-iree-test vjp-rules/iree-matches-eval-graph
    "(lambda (x y) ...): broadcast-in-dim → mul / div / sub → select → tanh / exp →
transpose → reshape → reduce-sum を通す関数の vjp-graph（全入力と、x だけの2通り）が、
IREE 実行で eval-graph と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 2)
        (graph (nb:trace-to-graph
                (nb:with-tracing (x y)
                  (let* ((wide (nb:broadcast-in-dim y '(2 3) '(1)))
                         (z (- (tanh (* x wide)) (exp (/ x (+ (* wide wide) 1.0)))))
                         (picked (nb:where (< x wide) z (* z 2.0)))
                         (flat (nb:reshape (nb:transpose picked) '(6))))
                    (values (nb:reduce-sum flat :axes '(0)) flat)))
                (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(3) :f32)))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let ((arrays (list (make-random-array (make-array-spec '(2 3) :f32) :seed seed)
                                        (make-random-array (make-array-spec '(3) :f32) :seed (+ seed 1))
                                        ;; 出力の余接線: reduce-sum（スカラー）と flat（6要素）。
                                        (make-random-array (make-array-spec '() :f32) :seed (+ seed 2))
                                        (make-random-array (make-array-spec '(6) :f32) :seed (+ seed 3)))))
                      (and (%vjp-iree-matches-eval-graph-p backend (nb::vjp-graph graph) arrays)
                           (%vjp-iree-matches-eval-graph-p backend (nb::vjp-graph graph :nonzero '(t nil))
                                                           arrays))))
                  :regression-id vjp-rules/iree-matches-eval-graph
                  :regression-file (regression-path "iree-vjp-rules-matches-eval-graph" :package "NABLA.IREE.TESTS")))))

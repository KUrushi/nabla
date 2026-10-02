;;;; 形状・縮約・dot-general の jvp を変換した graph の IREE 経由 end-to-end
;;;; テスト（issue #81）。
;;;;
;;;; 5 つの線形ルール（reshape / broadcast-in-dim / transpose / reduce-sum /
;;;; dot-general）と reduce-max（指示関数: compare / select / broadcast / 除算を
;;;; 含む）を1つの関数に通し、jvp-graph の結果が emit-stablehlo → IREE で
;;;; eval-graph と一致する。%jvp-iree-matches-eval-graph-p は jvp-test.lisp のもの。

(in-package #:nabla.iree.tests)

(define-iree-test jvp-shape/iree-matches-eval-graph
    "(lambda (a b) ...): dot → reduce-max → broadcast-in-dim → transpose → reshape →
reduce-sum を通す関数の jvp-graph（全接線と、a だけの接線の2通り）が、IREE 実行で
eval-graph と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 4)
        (graph (nb:trace-to-graph
                (nb:with-tracing (a b)
                  (let* ((d (nb:dot a b))
                         (m (nb:reduce-max d :axes '(1)))
                         (wide (nb:broadcast-in-dim m '(3 2) '(1)))
                         (flat (nb:reshape (nb:transpose wide) '(6))))
                    (values (nb:reduce-sum flat :axes '(0)) flat (nb:reduce-max d :axes '(0 1)))))
                (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(3 4) :f32)))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let ((arrays (list (make-random-array (make-array-spec '(2 3) :f32) :seed seed)
                                        (make-random-array (make-array-spec '(3 4) :f32) :seed (+ seed 1))
                                        (make-random-array (make-array-spec '(2 3) :f32) :seed (+ seed 2))
                                        (make-random-array (make-array-spec '(3 4) :f32) :seed (+ seed 3)))))
                      (and (%jvp-iree-matches-eval-graph-p backend (nb::jvp-graph graph) arrays)
                           (%jvp-iree-matches-eval-graph-p backend (nb::jvp-graph graph :nonzero '(t nil))
                                                           (subseq arrays 0 3)))))
                  :regression-id jvp-shape/iree-matches-eval-graph
                  :regression-file (regression-path "iree-jvp-shape-matches-eval-graph" :package "NABLA.IREE.TESTS")))))

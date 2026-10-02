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

(define-iree-test vjp-rules/dot-general-iree-matches-eval-graph
    "2層 MLP の損失 reduce-sum((tanh(x·W1 + b1)·W2 - y)^2) と、バッチ付きの dot-general
（バッチ次元と縮約次元が先頭でない）を通す関数の vjp-graph が、IREE 実行で eval-graph と一致する。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (*num-trials* 2)
         (mlp-shapes '((4 3) (4 2) (3 5) (5) (5 2)))
         (mlp (nb:trace-to-graph
               (nb:with-tracing (x y w1 b1 w2)
                 (let* ((h (tanh (+ (nb:dot x w1) (nb:broadcast-in-dim b1 '(4 5) '(1)))))
                        (diff (- (nb:dot h w2) y)))
                   (nb:reduce-sum (* diff diff) :axes '(0 1))))
               (mapcar (lambda (s) (nb:make-aval s :f32)) mlp-shapes)))
         (batched (nb:trace-to-graph
                   (nb:with-tracing (a b)
                     (nb::%trace-eqn :dot-general (list a b)
                                     :lhs-contracting '(0) :rhs-contracting '(2)
                                     :lhs-batch '(2) :rhs-batch '(0)))
                   (list (nb:make-aval '(3 2 4) :f32) (nb:make-aval '(4 5 3) :f32)))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (flet ((rnd (shape k) (make-random-array (make-array-spec shape :f32) :seed (+ seed k))))
                      (and (%vjp-iree-matches-eval-graph-p
                            backend (nb::vjp-graph mlp)
                            (append (loop for shape in mlp-shapes for k from 0 collect (rnd shape k))
                                    (list (rnd '() 9))))
                           (%vjp-iree-matches-eval-graph-p
                            backend (nb::vjp-graph batched)
                            (list (rnd '(3 2 4) 0) (rnd '(4 5 3) 1) (rnd '(4 2 5) 2))))))
                  :regression-id vjp-rules/dot-general-iree-matches-eval-graph
                  :regression-file (regression-path "iree-vjp-rules-dot-matches-eval-graph"
                                                    :package "NABLA.IREE.TESTS")))))

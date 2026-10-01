;;;; symbolic zero・instantiate-zero・add-tangents の性質（issue #77、77a）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %all-zero-p (array)
  "ARRAY の全要素が 0 か（bf16 / f16 のビット列でも 0 は全ビット 0）。"
  (dotimes (i (array-total-size array) t)
    (unless (zerop (row-major-aref array i)) (return nil))))

(defun %spec-aval (spec)
  "ARRAY-SPEC から同じ shape・dtype の aval を作る。"
  (nb:make-aval (array-spec-shape spec) (array-spec-dtype spec)))

(test ad-zero/struct-basics
  "make-symbolic-zero は aval を持ち、symbolic-zero-p は zero だけが真。
undefined-primal は aval を持つ別の型。"
  (let* ((aval (nb:make-aval '(2 3) :f32))
         (zero (nb::make-symbolic-zero aval))
         (undef (nb::make-undefined-primal aval)))
    (is (nb::symbolic-zero-p zero))
    (is (not (nb::symbolic-zero-p undef)))
    (is (not (nb::symbolic-zero-p 0.0)))
    (is (nb::undefined-primal-p undef))
    (is (not (nb::undefined-primal-p zero)))
    (is (equalp aval (nb::symbolic-zero-aval zero)))
    (is (equalp aval (nb::undefined-primal-aval undef)))
    (is (equalp aval (nb::tangent-aval zero)))))

(test ad-zero/tangent-aval-of-tracer
  "tangent-aval はトレーサの aval を返す。"
  (let* ((aval (nb:make-aval '(4) :f64))
         (seen nil)
         (recorder (lambda (x) (setf seen (nb::tangent-aval x)) x)))
    (nb:trace-to-graph (nb:with-tracing (x) (funcall recorder x)) (list aval))
    (is (equalp aval seen))))

(test ad-zero/instantiate-zero-evals-to-zeros
  "任意の aval の symbolic zero を with-tracing の中で instantiate-zero した
graph を eval-graph すると、その aval の shape・dtype を持つゼロ配列になる。"
  (is (check-it (generator (array-spec :dtypes *dtypes* :max-rank 3 :max-dim 4))
                (lambda (spec)
                  (let* ((aval (%spec-aval spec))
                         (zero (nb::make-symbolic-zero aval))
                         (graph (nb:trace-to-graph
                                 (nb:with-tracing () (nb::instantiate-zero zero))
                                 '()))
                         (result (nb:eval-graph graph)))
                    (and (equal (array-dimensions result) (array-spec-shape spec))
                         (equal (array-element-type result) (nabla.tests.support::element-type-for-dtype (array-spec-dtype spec)))
                         (%all-zero-p result))))
                :regression-id ad-zero/instantiate-zero-evals-to-zeros
                :regression-file (regression-path "ad-zero-instantiate-zero"))))

(test ad-zero/instantiate-zero-eqn-shape
  "rank 0 なら定数だけで eqn は足さず、rank 1 以上なら broadcast-in-dim が1つ
増える（%lift-number と同じ形）。"
  (flet ((graph-of (shape)
           (let ((zero (nb::make-symbolic-zero (nb:make-aval shape :f32))))
             (nb:trace-to-graph (nb:with-tracing () (nb::instantiate-zero zero)) '()))))
    (is (string= "(graph
 (:in)
 (:const (%0 f32 () 0.0))
 (:eqns)
 (:out %0))"
                 (nb:print-graph (graph-of '()))))
    (is (string= "(graph
 (:in)
 (:const (%0 f32 () 0.0))
 (:eqns
  (%1 f32 (2 3) := broadcast-in-dim (:shape (2 3) :dims ()) %0))
 (:out %1))"
                 (nb:print-graph (graph-of '(2 3)))))))

(test ad-zero/instantiate-zero-passes-tracers-through
  "トレーサに instantiate-zero を適用してもそのまま返り、eqn は増えない。"
  (let ((graph (nb:trace-to-graph (nb:with-tracing (x) (nb::instantiate-zero x))
                                  (list (nb:make-aval '(2) :f32)))))
    (is (null (nb:graph-eqns graph)))
    (is (eq (first (nb:graph-invars graph)) (first (nb:graph-outvars graph))))))

(test ad-zero/add-tangents-zero-is-identity
  "add-tangents は zero + t と t + zero で t をそのまま返し（eqn を足さない）、
zero + zero は zero を返す。"
  (is (check-it (generator (array-spec :dtypes *dtypes* :max-rank 3 :max-dim 4))
                (lambda (spec)
                  (let* ((aval (%spec-aval spec))
                         (zero (nb::make-symbolic-zero aval))
                         (left (nb:trace-to-graph
                                (nb:with-tracing (x) (nb::add-tangents zero x)) (list aval)))
                         (right (nb:trace-to-graph
                                 (nb:with-tracing (x) (nb::add-tangents x zero)) (list aval))))
                    (and (null (nb:graph-eqns left))
                         (null (nb:graph-eqns right))
                         (eq (first (nb:graph-invars left)) (first (nb:graph-outvars left)))
                         (eq (first (nb:graph-invars right)) (first (nb:graph-outvars right)))
                         (let ((z (nb::add-tangents zero zero)))
                           (and (nb::symbolic-zero-p z) (equalp aval (nb::symbolic-zero-aval z)))))))
                :regression-id ad-zero/add-tangents-zero-is-identity
                :regression-file (regression-path "ad-zero-add-tangents-identity"))))

(test ad-zero/add-tangents-nonzero-adds
  "どちらもゼロでなければ、graph は add の eqn を1つ持ち、eval-graph の結果は a + b。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f32 :f64) :max-rank 3 :max-dim 4)
                                  (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((aval (%spec-aval spec))
                           (a (make-random-array spec :seed seed))
                           (b (make-random-array spec :seed (1+ seed)))
                           (graph (nb:trace-to-graph
                                   (nb:with-tracing (x y) (nb::add-tangents x y)) (list aval aval)))
                           (result (nb:eval-graph graph a b)))
                      (and (= 1 (length (nb:graph-eqns graph)))
                           (eq :add (nb:primitive-name (nb:eqn-prim (first (nb:graph-eqns graph)))))
                           (allclose result (nb::%t-add a b) :dtype (array-spec-dtype spec))))))
                :regression-id ad-zero/add-tangents-nonzero-adds
                :regression-file (regression-path "ad-zero-add-tangents-add"))))

;;;; stop-gradient プリミティブ（issue #80）: 値は恒等、jvp（接線）は symbolic zero。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(test stop-gradient/abstract-eval-is-identity-for-any-dtype-and-shape
  "出力の aval は入力の aval そのまま（:i1 を含む任意の dtype・shape）。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (let* ((shape (loop for k from 1 to (mod seed 4) collect (mod (+ seed k) 4)))
                         (dtype (nth (mod seed 5) '(:f32 :f64 :bf16 :f16 :i1)))
                         (aval (nb:make-aval shape dtype)))
                    (equalp aval (funcall (nb::primitive-abstract-eval (nb::find-primitive :stop-gradient))
                                          (list aval)))))
                :regression-id stop-gradient/abstract-eval-is-identity-for-any-dtype-and-shape
                :regression-file (regression-path "stop-gradient-abstract-eval"))))

(test stop-gradient/eager-returns-an-equal-fresh-array
  "配列に適用すると、中身が等しく元と別の配列を返す（元を書き換えても影響しない）。"
  (let* ((x (make-random-array (make-array-spec '(2 3) :f32) :seed 3))
         (y (nb:stop-gradient x)))
    (is (equalp x y))
    (is (not (eq x y)))
    (is (equal '(2 3) (array-dimensions y)))
    (let ((bits (make-array '(3) :element-type 'bit :initial-contents '(1 0 1))))
      (is (equalp bits (nb:stop-gradient bits))))))

(test stop-gradient/tracer-adds-one-eqn
  "トレース中の (stop-gradient x) は :stop-gradient の eqn を1つ足し、eval-graph は恒等。"
  (let* ((aval (nb:make-aval '(3) :f64))
         (graph (nb:trace-to-graph (nb:with-tracing (x) (nb:stop-gradient x)) (list aval)))
         (x (make-random-array (make-array-spec '(3) :f64) :seed 1)))
    (is (equal '(:stop-gradient)
               (mapcar (lambda (e) (nb:primitive-name (nb:eqn-prim e))) (nb:graph-eqns graph))))
    (is (equalp x (nb:eval-graph graph x)))))

(test stop-gradient/jvp-tangent-is-zero
  "stop-gradient を通した値の接線はゼロ。(+ (stop-gradient x) x) の接線は x の接線そのもの。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (let* ((shape (loop for k from 1 to (mod seed 3) collect (1+ (mod (+ seed k) 3))))
                         (aval (nb:make-aval shape :f64))
                         (x (make-random-array (make-array-spec shape :f64) :seed seed))
                         (v (make-random-array (make-array-spec shape :f64) :seed (+ seed 1)))
                         (zero-graph (nb:trace-to-graph (nb:with-tracing (x) (nb:stop-gradient (* x x)))
                                                        (list aval)))
                         (sum-graph (nb:trace-to-graph (nb:with-tracing (x) (+ (nb:stop-gradient x) x))
                                                       (list aval)))
                         (zero-result (multiple-value-list (nb:eval-graph (nb::jvp-graph zero-graph) x v)))
                         (sum-result (multiple-value-list (nb:eval-graph (nb::jvp-graph sum-graph) x v))))
                    (and (every #'zerop (make-array (array-total-size x)
                                                    :displaced-to (second zero-result)
                                                    :element-type 'double-float))
                         (equalp v (second sum-result)))))
                :regression-id stop-gradient/jvp-tangent-is-zero
                :regression-file (regression-path "stop-gradient-jvp"))))

(test stop-gradient/emit-uses-optimization-barrier
  "StableHLO には恒等の op が無いので optimization_barrier を出す（docs/stablehlo-ops.md）。"
  (let* ((aval (nb:make-aval '(4) :f32))
         (graph (nb:trace-to-graph (nb:with-tracing (x) (nb:stop-gradient x)) (list aval)))
         (text (nb:emit-stablehlo graph)))
    (is (search "stablehlo.optimization_barrier" text))))

(test stop-gradient/abstract-eval-rejects-wrong-arity
  "入力が1つでなければ primitive-error。"
  (let ((aval (nb:make-aval '(2) :f32))
        (rule (nb::primitive-abstract-eval (nb::find-primitive :stop-gradient))))
    (signals nb:primitive-error (funcall rule (list aval aval)))
    (signals nb:primitive-error (funcall rule '()))))

;;;; nb:grad / nb:value-and-grad の :has-aux（issue #260）。
;;;;
;;;; f は (values 損失 aux) を返し、aux（PyTree）には勾配を流さない（JAX の has_aux）。
;;;; 期待値は :has-aux を使わない value-and-grad（損失だけを返す同じ関数）と、eager に評価した aux。
;;;; BatchNorm の学習ステップで、更新後のモジュールを aux として取り出すのに使う。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %aux-function (graph)
  "GRAPH の最初の出力の reduce-sum を損失、(GRAPH の全出力 7 nil) を aux とする TRACEABLE-FUNCTION。
aux はトレーサ・Lisp の整数・空の部分木を混ぜた木。"
  (nb::%make-traceable-function
   (mapcar (lambda (v) (declare (ignore v)) (gensym "X")) (nb:graph-invars graph))
   (lambda (&rest tracers)
     (let ((outs (nb::inline-graph graph tracers)))
       (values (nb:reduce-sum (first outs)) (list outs 7 nil))))))

(test grad-has-aux/matches-value-and-grad-without-aux-and-eager-aux
  "ランダムな graph で、(value-and-grad f :has-aux t) の値と勾配は aux を返さない同じ損失の
value-and-grad と一致し、aux は graph を eager に評価した出力（と素通しの 7 と nil）に一致する。
(grad f :has-aux t) は (values 勾配 aux) を返す。"
  (let ((*primitive-recipe-dtypes* '(:f64 :f32)))
    (is (check-it (generator (primitive-graph-recipe :max-ops 4))
                  (lambda (recipe)
                    (let* ((graph (build-primitive-graph recipe))
                           (arrays (%jvp-arrays graph :seed (%recipe-seed recipe)))
                           (argnums (loop for i below (length arrays) collect i))
                           (float-p (every (lambda (a) (member (%array-dtype a) '(:f32 :f64))) arrays))
                           (expected-aux (multiple-value-list (apply #'nb:eval-graph graph arrays))))
                      (or (not float-p)
                          (not (member (%array-dtype (first expected-aux)) '(:f32 :f64)))
                          (multiple-value-bind (value grads)
                              (apply (nb:value-and-grad (scalar-loss-function graph) :argnums argnums) arrays)
                            (multiple-value-bind (value-a grads-a aux)
                                (apply (nb:value-and-grad (%aux-function graph) :argnums argnums :has-aux t)
                                       arrays)
                              (multiple-value-bind (grads-g aux-g)
                                  (apply (nb:grad (%aux-function graph) :argnums argnums :has-aux t) arrays)
                                (and (equalp value value-a)
                                     (equalp grads grads-a)
                                     (equalp grads grads-g)
                                     (= 3 (length aux))
                                     (eql 7 (second aux))
                                     (null (third aux))
                                     (%results-close-p (first aux) expected-aux :rtol 0 :atol 0)
                                     (equalp aux aux-g))))))))
                  :regression-id grad-has-aux/matches-value-and-grad-without-aux-and-eager-aux
                  :regression-file (regression-path "grad-has-aux")))))

(test grad-has-aux/aux-is-differentiable-by-an-outer-grad
  "aux は内側の grad では微分しないが、外側の変換からは普通の値（JAX と同じ）。
f(x) = (values sum(x^3) x^2) のとき内側の勾配は 3x^2、aux は x^2 で、
h(x) = sum(3x^2) + sum(x^2) = 4 sum(x^2) の勾配は 8x。"
  (let* ((f (nb:with-tracing (x) (values (nb:reduce-sum (* x x x)) (* x x))))
         (g (nb:grad f :has-aux t))
         (h (nb:with-tracing (x)
              (multiple-value-bind (dx aux) (funcall g x)
                (+ (nb:reduce-sum dx) (nb:reduce-sum aux)))))
         (x (%vec 1d0 -2d0 0.5d0)))
    (multiple-value-bind (dx aux) (funcall g x)
      (is (equalp (%vec 3d0 12d0 0.75d0) dx))
      (is (equalp (%vec 1d0 4d0 0.25d0) aux)))
    (is (equalp (%vec 8d0 -16d0 4d0) (funcall (nb:grad h) x)))))

(test grad-has-aux/non-float-and-host-leaves-pass-through
  ":i1 のトレーサ・:i32 のトレーサ・ホストの配列・Lisp の値の葉も aux に入れられる。
ホストの値はそのまま（EQ）返り、トレーサの葉は eager に評価した値になる。"
  (let* ((host (%vec 9d0 8d0))
         (f (nb:with-tracing (x n)
              (values (nb:reduce-sum (* x x))
                      (vector (< x 0d0) (+ n 1) host :tag))))
         (x (%vec -1d0 2d0))
         (n (make-array '() :element-type '(signed-byte 32) :initial-element 4)))
    (multiple-value-bind (value grad aux) (funcall (nb:value-and-grad f :has-aux t) x n)
      (is (equalp (%f64-scalar 5d0) value))
      (is (equalp (%vec -2d0 4d0) grad))
      (is (equalp (make-array '(2) :element-type 'bit :initial-contents '(1 0)) (svref aux 0)))
      (is (equalp (make-array '() :element-type '(signed-byte 32) :initial-element 5) (svref aux 1)))
      (is (eq host (svref aux 2)))
      (is (eq :tag (svref aux 3))))))

(test grad-has-aux/works-inside-a-trace
  "with-tracing の本体の中で (value-and-grad f :has-aux t) を使った graph を eval-graph した結果は
eager と一致する（aux のトレーサは外側のトレースの値になる）。"
  (let* ((f (nb:with-tracing (x) (values (nb:reduce-sum (tanh x)) (list (* 2d0 x) (exp x)))))
         (vg (nb:value-and-grad f :has-aux t))
         (outer (nb:with-tracing (x)
                  (multiple-value-bind (v g aux) (funcall vg x)
                    (values v g (first aux) (second aux)))))
         (graph (nb:trace-to-graph outer (list (nb:make-aval '(3) :f64))))
         (x (%vec 0.5d0 -1d0 2d0)))
    (is (%results-close-p (multiple-value-list (nb:eval-graph graph x))
                          (multiple-value-list (funcall outer x))
                          :rtol 0 :atol 0))
    (multiple-value-bind (v g aux) (funcall vg x)
      (declare (ignore v g))
      (is (equalp (%vec 1d0 -2d0 4d0) (first aux))))))

(test grad-has-aux/wrong-output-shape-signals-grad-requires-scalar-output
  ":has-aux の f は (values 損失 aux) の2値を返す。1値・3値、損失が木で包まれている・rank 0 でない
ときは grad-requires-scalar-output。:has-aux が T / NIL でなければ作る時点で autodiff-error。"
  (let ((x (%vec 1d0 2d0)))
    (dolist (f (list (nb:with-tracing (x) (nb:reduce-sum x))
                     (nb:with-tracing (x) (values (nb:reduce-sum x) x x))
                     (nb:with-tracing (x) (values (list (nb:reduce-sum x)) x))
                     (nb:with-tracing (x) (values x x))))
      (signals nb:grad-requires-scalar-output (funcall (nb:grad f :has-aux t) x))
      (signals nb:grad-requires-scalar-output (funcall (nb:value-and-grad f :has-aux t) x)))
    (signals nb:autodiff-error (nb:grad (nb:with-tracing (x) x) :has-aux :yes))
    (signals nb:autodiff-error (nb:value-and-grad (nb:with-tracing (x) x) :has-aux 1))))

(test grad-has-aux/while-loop-depending-on-a-differentiated-argument-is-rejected
  "既知の制限: 微分する引数に依存する while-loop の結果は aux に入れられない（autodiff-error）。
while-loop が微分しない引数だけに依存するなら通る。"
  (let ((cond-fn (nb:with-tracing (c) (< (nb:reduce-sum c) 100d0)))
        (body-fn (nb:with-tracing (c) (* 2d0 c)))
        (x (%vec 1d0 -2d0 0.5d0)))
    (signals nb:autodiff-error
      (funcall (nb:grad (nb:with-tracing (x)
                          (values (nb:reduce-sum (* x x)) (nb:while-loop cond-fn body-fn (* x x))))
                        :has-aux t)
               x))
    (multiple-value-bind (dx aux)
        (funcall (nb:grad (nb:with-tracing (x y)
                            (values (nb:reduce-sum (* x x)) (nb:while-loop cond-fn body-fn y)))
                          :has-aux t)
                 x (%vec 1d0 2d0))
      (is (equalp (%vec 2d0 -4d0 1d0) dx))
      (is (equalp (%vec 64d0 128d0) aux)))))

(test grad-has-aux/vjp-graph-n-cotangents
  "vjp-graph の :n-cotangents（内部）: m（出力の個数）なら既定と同じ結果、0 なら余接線を取らず
入力の余接線は 0、範囲外・整数でないと autodiff-error。2出力の graph (x*y, x+y) で確かめる。"
  (let* ((graph (nb:trace-to-graph (nb:with-tracing (x y) (values (* x y) (+ x y)))
                                   (list (nb:make-aval '() :f64) (nb:make-aval '() :f64))))
         (x (%f64-scalar 3d0)) (y (%f64-scalar 5d0))
         (u (%f64-scalar 2d0)) (v (%f64-scalar 7d0)))
    (is (equalp (multiple-value-list (nb:eval-graph (nb::vjp-graph graph) x y u v))
                (multiple-value-list (nb:eval-graph (nb::vjp-graph graph :n-cotangents 2) x y u v))))
    ;; 先頭の出力だけ: d(x*y)·u = (y u, x u)
    (is (equalp (list (%f64-scalar 15d0) (%f64-scalar 8d0) (%f64-scalar 10d0) (%f64-scalar 6d0))
                (multiple-value-list (nb:eval-graph (nb::vjp-graph graph :n-cotangents 1) x y u))))
    (is (equalp (list (%f64-scalar 15d0) (%f64-scalar 8d0) (%f64-scalar 0d0) (%f64-scalar 0d0))
                (multiple-value-list (nb:eval-graph (nb::vjp-graph graph :n-cotangents 0) x y))))
    (dolist (bad '(3 -1 1.0 :x))
      (signals nb:autodiff-error (nb::vjp-graph graph :n-cotangents bad)))))

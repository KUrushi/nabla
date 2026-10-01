;;;; tests/support/autodiff.lisp（自動微分のテスト支援、issue #76）に対する PBT。
;;;;
;;;; 後続のルール（#77, #80–#84, #86）のテストが信用する道具なので、
;;;; 解析的に分かる例と、双線形性のような代数的な性質で確かめる。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %f64-array-like (shape fill)
  (make-array shape :element-type 'double-float :initial-element fill))

(defun %poly (x a b c)
  "要素ごとの多項式 a x^3 + b x^2 + c x（f64 配列 X）。"
  (let ((out (%f64-array-like (array-dimensions x) 0d0)))
    (dotimes (i (array-total-size x) out)
      (let ((v (row-major-aref x i)))
        (setf (row-major-aref out i) (+ (* a v v v) (* b v v) (* c v)))))))

(defun %poly-derivative (x a b c)
  "上の多項式の要素ごとの導関数 3 a x^2 + 2 b x + c。"
  (let ((out (%f64-array-like (array-dimensions x) 0d0)))
    (dotimes (i (array-total-size x) out)
      (let ((v (row-major-aref x i)))
        (setf (row-major-aref out i) (+ (* 3 a v v) (* 2 b v) c))))))

(defun %elementwise-product (p q)
  (let ((out (%f64-array-like (array-dimensions p) 0d0)))
    (dotimes (i (array-total-size p) out)
      (setf (row-major-aref out i) (* (row-major-aref p i) (row-major-aref q i))))))

(test support/autodiff/central-difference-matches-analytic-polynomial
  "解析的に微分が分かる多項式 a x^3 + b x^2 + c x について、中心差分の方向微分
（f'(x) * v）と勾配（余接線 u との縮約 u * f'(x)）が、解析解と
*AUTODIFF-RTOL* / *AUTODIFF-ATOL* で一致する。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f64) :max-rank 3 :max-dim 4)
                                  (uniform-integer :lo 0 :hi (1- (expt 2 31)))
                                  (uniform-real :lo -2d0 :hi 2d0)
                                  (uniform-real :lo -2d0 :hi 2d0)
                                  (uniform-real :lo -2d0 :hi 2d0)))
                (lambda (case)
                  (destructuring-bind (spec seed a b c) case
                    (let* ((a (coerce a 'double-float))
                           (b (coerce b 'double-float))
                           (c (coerce c 'double-float))
                           (aval (nb:make-aval (array-spec-shape spec) :f64))
                           (x (make-random-array spec :seed seed))
                           (v (random-tangent aval :seed (+ seed 1)))
                           (u (random-cotangent aval :seed (+ seed 2)))
                           (fn (lambda (x) (%poly x a b c)))
                           (deriv (%poly-derivative x a b c))
                           (jvp (first (central-difference-jvp fn (list x) (list v))))
                           (grad (first (central-difference-gradient
                                         fn (list x) :cotangents (list u)))))
                      (and (allclose jvp (%elementwise-product deriv v)
                                     :rtol *autodiff-rtol* :atol *autodiff-atol*)
                           (allclose grad (%elementwise-product deriv u)
                                     :rtol *autodiff-rtol* :atol *autodiff-atol*)))))
                :regression-id support/autodiff/central-difference-matches-analytic-polynomial
                :regression-file (regression-path "autodiff-central-difference-polynomial"))))

(test support/autodiff/central-difference-evaluates-graphs
  "graph も受け取れる: x^3 を graph（mul を2回）で作ると、x = 2 での微分は 12。
2入力（x と y の積）の勾配は入力ごとに返る。"
  (let* ((x (nb::make-var (nb:make-aval '() :f64)))
         (sq (nb::make-eqn :mul (list x x)))
         (cube (nb::make-eqn :mul (list (first (nb:eqn-outvars sq)) x)))
         (graph (nb::make-graph (list x) (list sq cube) (nb:eqn-outvars cube)))
         (two (%f64-array-like '() 2d0))
         (one (%f64-array-like '() 1d0)))
    (is (nb::check-graph graph))
    (is (approx= (aref (first (central-difference-jvp graph (list two) (list one))))
                 12d0 :rtol *autodiff-rtol* :atol *autodiff-atol*))
    (is (approx= (aref (first (central-difference-gradient graph (list two))))
                 12d0 :rtol *autodiff-rtol* :atol *autodiff-atol*)))
  (let* ((x (nb::make-var (nb:make-aval '(2) :f64)))
         (y (nb::make-var (nb:make-aval '(2) :f64)))
         (prod (nb::make-eqn :mul (list x y)))
         (graph (nb::make-graph (list x y) (list prod) (nb:eqn-outvars prod)))
         (xa (make-array 2 :element-type 'double-float :initial-contents '(2d0 3d0)))
         (ya (make-array 2 :element-type 'double-float :initial-contents '(5d0 7d0)))
         (grads (central-difference-gradient graph (list xa ya))))
    (is (allclose (first grads) ya :rtol *autodiff-rtol* :atol *autodiff-atol*))
    (is (allclose (second grads) xa :rtol *autodiff-rtol* :atol *autodiff-atol*))))

(test support/autodiff/inner-product-is-bilinear
  "INNER-PRODUCT は双線形（第1・第2引数それぞれで線形）で対称で、
独立に書いた要素ごとの和と一致する。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f64) :max-rank 4 :max-dim 4)
                                  (uniform-integer :lo 0 :hi (1- (expt 2 31)))
                                  (uniform-real :lo -3d0 :hi 3d0)
                                  (uniform-real :lo -3d0 :hi 3d0)))
                (lambda (case)
                  (destructuring-bind (spec seed s r) case
                    (let* ((s (coerce s 'double-float))
                           (r (coerce r 'double-float))
                           (aval (nb:make-aval (array-spec-shape spec) :f64))
                           (a (random-tangent aval :seed seed))
                           (a2 (random-tangent aval :seed (+ seed 1)))
                           (b (random-tangent aval :seed (+ seed 2)))
                           (comb (%f64-array-like (array-dimensions a) 0d0))
                           (sum 0d0))
                      (dotimes (i (array-total-size a))
                        (setf (row-major-aref comb i)
                              (+ (* s (row-major-aref a i)) (* r (row-major-aref a2 i))))
                        (incf sum (* (row-major-aref a i) (row-major-aref b i))))
                      (flet ((close-p (x y) (approx= x y :rtol 1d-12 :atol 1d-12)))
                        (and (close-p (inner-product comb b)
                                      (+ (* s (inner-product a b)) (* r (inner-product a2 b))))
                             (close-p (inner-product b comb)
                                      (+ (* s (inner-product b a)) (* r (inner-product b a2))))
                             (close-p (inner-product a b) (inner-product b a))
                             (close-p (inner-product a b) sum))))))
                :regression-id support/autodiff/inner-product-is-bilinear
                :regression-file (regression-path "autodiff-inner-product-bilinear"))))

(test support/autodiff/inner-product-rejects-shape-mismatch
  "形が違う配列の内積はエラー。"
  (signals error (inner-product (%f64-array-like '(2) 1d0) (%f64-array-like '(3) 1d0))))

(test support/autodiff/random-tangent-is-deterministic-f64
  "RANDOM-TANGENT は aval と同じ形の f64 配列で、同じ seed なら同じ、別の seed なら別。"
  (let* ((aval (nb:make-aval '(3 2) :f32))
         (a (random-tangent aval :seed 7)))
    (is (equal (array-dimensions a) '(3 2)))
    (is (eq (array-element-type a) 'double-float))
    (is (equalp a (random-tangent aval :seed 7)))
    (is (equalp a (random-cotangent aval :seed 7)))
    (is (not (equalp a (random-tangent aval :seed 8))))))

(test support/autodiff/f64-recipe-graphs-are-valid
  "*PRIMITIVE-RECIPE-DTYPES* を '(:f64) に束縛すると、f64 だけのランダムな graph を
作れ、CHECK-GRAPH を通り、EVAL-GRAPH で評価できる。"
  (let ((*primitive-recipe-dtypes* '(:f64)))
    (is (check-it (generator (primitive-graph-recipe :max-ops 4))
                  (lambda (recipe)
                    (let* ((graph (build-primitive-graph recipe))
                           (args (mapcar (lambda (v)
                                           (make-random-array
                                            (make-array-spec (nb:aval-shape (nb:var-aval v)) :f64)
                                            :seed 1))
                                         (nb:graph-invars graph))))
                      (and (nb::check-graph graph)
                           (every (lambda (v) (eq (nb:aval-dtype (nb:var-aval v)) :f64))
                                  (append (nb:graph-invars graph) (nb:graph-outvars graph)))
                           (progn (apply #'nb:eval-graph graph args) t))))
                  :regression-id support/autodiff/f64-recipe-graphs-are-valid
                  :regression-file (regression-path "autodiff-f64-recipe-graphs")))))

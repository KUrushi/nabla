;;;; 線形プリミティブの transpose ルールの性質（issue #83）。
;;;;
;;;; 対象: sub convert reshape transpose broadcast-in-dim reduce-sum select
;;;; と、片側が既知の mul / div（add / neg は #82、dot-general は #84）。
;;;; 各ルールを「線形な graph（既知の入力 ++ 線形入力）を1つ作り、
;;;; nb::transpose-graph で転置する」形で確かめる。守らせる性質:
;;;;   1. 随伴性: <T(u), v> = <u, L(v)>（L は元の graph を eval-graph したもの）
;;;;   2. T の線形性: T(a·u + b·w) = a·T(u) + b·T(w)
;;;;   3. 内積テスト: 実プリミティブのランダムな f64 の graph（非線形の
;;;;      exp / tanh / max / min / reduce-max / compare+select も含む。dot-general は
;;;;      #84 まで除く）について <vjp(u), v> = <u, jvp(v)>
;;;; 加えて、固定の例（broadcast-in-dim のサイズ1の広がりなど）と、線形でない
;;;; 使い方・stop-gradient の扱いを確かめる。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %tr-graph (avals fn)
  (nb::%call-with-fresh-trace avals fn))

(defun %f64-avals (n shape) (loop repeat n collect (nb:make-aval shape :f64)))

(defun %tr-case (seed)
  "SEED から、(VALUES graph n-known 近似) を作る。graph の入力は既知の N-KNOWN 個に
線形入力が続く。近似が真なら f32 を通る（許容誤差を緩める）。"
  (let* ((kind (mod seed 11))
         (rest (floor seed 11))
         (rank (mod rest 4))
         (shape (%seed-shape (floor rest 4) rank))
         (aval (nb:make-aval shape :f64))
         (total (reduce #'* shape)))
    (ecase kind
      (0 (values (%tr-graph (list aval aval) (lambda (a b) (nb::%trace-eqn :sub (list a b)))) 0 nil))
      (1 (let ((to (if (evenp rest) :f32 :f64))
               (from (if (evenp rest) :f64 :f32)))
           (values (%tr-graph (list (nb:make-aval shape from))
                              (lambda (a) (nb::%trace-eqn :convert (list a) :dtype to)))
                   0 t)))
      (2 (values (%tr-graph (list aval)
                            (lambda (a) (nb::%trace-eqn :reshape (list a)
                                                        :shape (if (and (plusp rank) (evenp rest))
                                                                   (list total 1)
                                                                   (list total)))))
                 0 nil))
      (3 (let ((perm (%shuffle-by-seed (loop for i below rank collect i) rest)))
           (values (%tr-graph (list aval) (lambda (a) (nb::%trace-eqn :transpose (list a) :perm perm)))
                   0 nil)))
      (4 (values (%broadcast-case-graph rest) 0 nil))
      (5 (let* ((rank (max 1 rank))
                (aval (nb:make-aval (%seed-shape (floor rest 4) rank) :f64))
                (axes (%seed-subset rest rank)))
           (values (%tr-graph (list aval) (lambda (a) (nb::%trace-eqn :reduce-sum (list a) :axes axes)))
                   0 nil)))
      ;; select: pred は既知の2入力の compare。値側は2つとも線形。
      (6 (values (%tr-graph (%f64-avals 4 shape)
                            (lambda (p q a b)
                              (nb::%trace-eqn :select
                                              (list (nb::%trace-eqn :compare (list p q) :direction :lt) a b))))
                 2 nil))
      ;; 片方の値だけが線形（もう片方は既知で、L が線形になるよう 0 = p - p にする）。
      (7 (values (%tr-graph (%f64-avals 3 shape)
                            (lambda (p q b)
                              (nb::%trace-eqn :select
                                              (list (nb::%trace-eqn :compare (list p q) :direction :gt)
                                                    (nb::%trace-eqn :sub (list p p))
                                                    b))))
                 2 nil))
      ;; mul: 既知の係数が左、右。
      (8 (values (%tr-graph (%f64-avals 2 shape) (lambda (y x) (nb::%trace-eqn :mul (list y x)))) 1 nil))
      (9 (values (%tr-graph (%f64-avals 2 shape) (lambda (y x) (nb::%trace-eqn :mul (list x y)))) 1 nil))
      ;; div: 既知の除数（exp p > 0）で線形な被除数を割る。
      (10 (values (%tr-graph (%f64-avals 2 shape)
                             (lambda (p x) (nb::%trace-eqn :div (list x (nb::%trace-eqn :exp (list p))))))
                  1 nil)))))

(defun %tr-tolerance (approximate) (if approximate 1d-5 1d-9))

(defun %tr-random-cotangents (graph seed)
  (loop for outvar in (nb:graph-outvars graph) for i from 0
        collect (let ((aval (nb:var-aval outvar)))
                  (random-cotangent aval :seed (+ seed 500 i) :dtype (nb:aval-dtype aval)))))

(defun %tr-known-and-linear (graph n-known seed)
  "(VALUES 既知の入力 線形入力 v)。"
  (values (subseq (%jvp-arrays graph :seed seed) 0 n-known)
          (nthcdr n-known (%jvp-arrays graph :tangent t :seed seed))))

(defun %tr-close-p (a b tolerance)
  (<= (abs (- a b)) (* tolerance (+ 1d0 (abs a) (abs b)))))

(defun %tr-well-formed-p (graph transposed n-known)
  (and (%jvp-round-trips-p transposed)
       (equalp (mapcar #'nb:var-aval (nb:graph-outvars transposed))
               (mapcar #'nb:var-aval (nthcdr n-known (nb:graph-invars graph))))
       (equalp (mapcar #'nb:var-aval (nb:graph-invars transposed))
               (append (mapcar #'nb:var-aval (subseq (nb:graph-invars graph) 0 n-known))
                       (mapcar #'nb:var-aval (nb:graph-outvars graph))))))

(test transpose-rules/adjoint
  "随伴性: <T(u), v> = <u, L(v)>。L は元の線形な graph（既知の入力は固定）、T は
transpose-graph の結果。sub / convert / reshape / transpose / broadcast-in-dim（ランダムな
dims・サイズ1・rank 0）/ reduce-sum（ランダムな axes）/ select / mul / div。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (multiple-value-bind (graph n-known approximate) (%tr-case seed)
                    (let ((transposed (nb::transpose-graph graph n-known)))
                      (multiple-value-bind (known v) (%tr-known-and-linear graph n-known seed)
                        (let* ((u (%tr-random-cotangents graph seed))
                               (l-v (%jvp-eval graph (append known v)))
                               (t-u (%jvp-eval transposed (append known u))))
                          (and (%tr-well-formed-p graph transposed n-known)
                               (%tr-close-p (%sum-inner-products t-u v) (%sum-inner-products u l-v)
                                            (%tr-tolerance approximate))))))))
                :regression-id transpose-rules/adjoint
                :regression-file (regression-path "transpose-rules-adjoint"))))

(test transpose-rules/transpose-is-linear
  "T の線形性: T(a·u + b·w) = a·T(u) + b·T(w)。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (multiple-value-bind (graph n-known approximate) (%tr-case seed)
                    (let* ((transposed (nb::transpose-graph graph n-known))
                           (known (%tr-known-and-linear graph n-known seed))
                           (u (%tr-random-cotangents graph seed))
                           (w (%tr-random-cotangents graph (+ seed 31)))
                           (a 2.5d0) (b -1.5d0)
                           (tolerance (%tr-tolerance approximate)))
                      (flet ((tt (cts) (%jvp-eval transposed (append known cts)))
                             (combine (x y) (%sum-array (%scale-array x a) (%scale-array y b))))
                        (%results-close-p (tt (mapcar #'combine u w))
                                          (mapcar #'combine (tt u) (tt w))
                                          :rtol tolerance :atol tolerance)))))
                :regression-id transpose-rules/transpose-is-linear
                :regression-file (regression-path "transpose-rules-linear"))))

(defun %finite-number-p (x)
  (and (realp x) (< (abs x) 1d100)))

(test transpose-rules/vjp-inner-product-identity-on-primitive-graphs
  "内積テスト: 実プリミティブ（add sub mul max min neg tanh exp compare+select convert
reshape transpose broadcast-in-dim reduce-sum reduce-max。dot-general は #84 まで除く）の
ランダムな f64 の graph について <vjp(u), v> = <u, jvp(v)>。主値の出力も一致する。
非有限の値が出た graph（exp の連鎖）は対象外。"
  (let ((*primitive-recipe-dtypes* '(:f64))
        (*primitive-recipe-dot-p* nil))
    (is (check-it (generator (primitive-graph-recipe :max-ops 6))
                  (lambda (recipe)
                    (let* ((seed (%recipe-seed recipe))
                           (graph (build-primitive-graph recipe))
                           (m (length (nb:graph-outvars graph)))
                           (primals (%jvp-arrays graph :seed seed))
                           (v (%jvp-arrays graph :tangent t :seed seed))
                           (u (%outputs-random-cotangents graph :seed seed))
                           (jvp-result (%jvp-eval (nb::jvp-graph graph) (append primals v)))
                           (vjp-result (%jvp-eval (nb::vjp-graph graph) (append primals u)))
                           (lhs (%sum-inner-products (nthcdr m vjp-result) v))
                           (rhs (%sum-inner-products u (nthcdr m jvp-result))))
                      (if (and (%finite-number-p lhs) (%finite-number-p rhs))
                          (and (%results-close-p (subseq vjp-result 0 m) (subseq jvp-result 0 m)
                                                 :rtol 1d-12 :atol 1d-12)
                               (%scalar-close-p lhs rhs))
                          t)))
                  :regression-id transpose-rules/vjp-inner-product-identity-on-primitive-graphs
                  :regression-file (regression-path "transpose-rules-vjp-inner-product")))))

;;; --- 固定の例 ---

(defun %vjp-cotangents (prim-name avals params arrays)
  "PRIM-NAME を PARAMS で1回適用する graph（入力 AVALS）の vjp の、入力の余接線のリスト。
ARRAYS は入力の主値 ++ 出力の余接線。"
  (let* ((vars (mapcar #'nb::make-var avals))
         (eqn (apply #'nb::make-eqn prim-name vars params))
         (graph (nb::make-graph vars (list eqn) (nb:eqn-outvars eqn) '())))
    (nthcdr 1 (%jvp-eval (nb::vjp-graph graph) arrays))))

(defun %f64 (shape &rest elements)
  (make-array shape :element-type 'double-float :initial-contents elements))

(test transpose-rules/broadcast-in-dim-fixed-examples
  "broadcast-in-dim の転置: 増えた軸、サイズ1から広げた軸を足し、rank 0 は全部足す。"
  (is (equalp (list (%f64 '(1 3) '(2d0 4d0 6d0)))
              (%vjp-cotangents :broadcast-in-dim (list (nb:make-aval '(1 3) :f64))
                               '(:shape (2 3) :dims (0 1))
                               (list (%f64 '(1 3) '(0d0 0d0 0d0))
                                     (%f64 '(2 3) '(1d0 2d0 3d0) '(1d0 2d0 3d0))))))
  (is (equalp (list (%f64 '(3) 5d0 7d0 9d0))
              (%vjp-cotangents :broadcast-in-dim (list (nb:make-aval '(3) :f64))
                               '(:shape (2 3) :dims (1))
                               (list (%f64 '(3) 0d0 0d0 0d0)
                                     (%f64 '(2 3) '(1d0 2d0 3d0) '(4d0 5d0 6d0))))))
  (is (equalp (list (make-array '() :element-type 'double-float :initial-element 21d0))
              (%vjp-cotangents :broadcast-in-dim (list (nb:make-aval '() :f64))
                               '(:shape (2 3) :dims ())
                               (list (make-array '() :element-type 'double-float :initial-element 0d0)
                                     (%f64 '(2 3) '(1d0 2d0 3d0) '(4d0 5d0 6d0)))))))

(test transpose-rules/convert-restores-the-input-dtype
  "convert の転置の余接線は元の入力の dtype（f32 → f64 の転置は f32）。"
  (let* ((x (nb::make-var (nb:make-aval '(2) :f32)))
         (eqn (nb::make-eqn :convert (list x) :dtype :f64))
         (graph (nb::make-graph (list x) (list eqn) (nb:eqn-outvars eqn) '()))
         (vjp (nb::vjp-graph graph)))
    (is (equalp (nb:make-aval '(2) :f32) (nb:var-aval (second (nb:graph-outvars vjp)))))))

(test transpose-rules/mul-and-div-with-linear-factors-signal-autodiff-error
  "mul で両方が線形、div で除数が線形な graph の転置は autodiff-error。"
  (dolist (prim '(:mul :div))
    (let* ((a (nb::make-var (nb:make-aval '(2) :f64)))
           (b (nb::make-var (nb:make-aval '(2) :f64)))
           (eqn (nb::make-eqn prim (list a b)))
           (graph (nb::make-graph (list a b) (list eqn) (nb:eqn-outvars eqn) '())))
      (signals nb:autodiff-error (nb::transpose-graph graph 0))))
  ;; 除数が線形（被除数が既知）。
  (let* ((x (nb::make-var (nb:make-aval '(2) :f64)))
         (y (nb::make-var (nb:make-aval '(2) :f64)))
         (eqn (nb::make-eqn :div (list x y)))
         (graph (nb::make-graph (list x y) (list eqn) (nb:eqn-outvars eqn) '())))
    (signals nb:autodiff-error (nb::transpose-graph graph 1))))

(test transpose-rules/stop-gradient-has-no-transpose-rule-but-vjp-is-zero
  "stop-gradient の jvp は symbolic zero なので、接線は線形側に流れず vjp の余接線は
ゼロ。線形入力に直接 stop-gradient がある graph の転置は no-transpose-rule（JAX と同じ。
手で作った graph でしか起きない）。"
  (let* ((x (nb::make-var (nb:make-aval '(2) :f64)))
         (eqn (nb::make-eqn :stop-gradient (list x)))
         (graph (nb::make-graph (list x) (list eqn) (nb:eqn-outvars eqn) '())))
    (is (equalp (list (%f64 '(2) 0d0 0d0))
                (nthcdr 1 (%jvp-eval (nb::vjp-graph graph) (list (%f64 '(2) 1d0 1d0) (%f64 '(2) 3d0 3d0))))))
    (signals nb:no-transpose-rule (nb::transpose-graph graph 0))))

(test transpose-rules/broadcast-in-dim-with-unsorted-dims
  "dims が昇順でない broadcast-in-dim（out[i, j] = x[j, i]）の転置は、余接線を転置して返す。"
  (is (equalp (list (%f64 '(2 3) '(1d0 3d0 5d0) '(2d0 4d0 6d0)))
              (%vjp-cotangents :broadcast-in-dim (list (nb:make-aval '(2 3) :f64))
                               '(:shape (3 2) :dims (1 0))
                               (list (%f64 '(2 3) '(0d0 0d0 0d0) '(0d0 0d0 0d0))
                                     (%f64 '(3 2) '(1d0 2d0) '(3d0 4d0) '(5d0 6d0)))))))

(test transpose-rules/select-with-linear-pred-signals-autodiff-error
  "select の pred は既知の主値でなければならない。pred が線形入力の graph の転置は autodiff-error。"
  (let* ((p (nb::make-var (nb:make-aval '(2) :i1)))
         (a (nb::make-var (nb:make-aval '(2) :f64)))
         (b (nb::make-var (nb:make-aval '(2) :f64)))
         (eqn (nb::make-eqn :select (list p a b)))
         (graph (nb::make-graph (list p a b) (list eqn) (nb:eqn-outvars eqn) '())))
    (signals nb:autodiff-error (nb::transpose-graph graph 0))))

(test transpose-rules/broadcast-in-dim-adds-only-the-needed-eqns
  "broadcast-in-dim の転置は、必要な eqn だけを足す: 増えた軸だけなら reduce-sum だけ
（余計な reshape / transpose を足さない）。"
  (let* ((x (nb::make-var (nb:make-aval '(3) :f64)))
         (eqn (nb::make-eqn :broadcast-in-dim (list x) :shape '(2 3) :dims '(1)))
         (graph (nb::make-graph (list x) (list eqn) (nb:eqn-outvars eqn) '()))
         (transposed (nb::transpose-graph graph 0)))
    (is (equal '(:reduce-sum)
               (mapcar (lambda (e) (nb:primitive-name (nb:eqn-prim e))) (nb:graph-eqns transposed))))))

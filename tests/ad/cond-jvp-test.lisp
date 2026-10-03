;;;; cond の jvp ルール・transpose ルール・linearize 経由の grad（issue #134）。
;;;;
;;;; 対象は x・w・v（どれも (3) の f64）を引数に取る cond*:
;;;;   pred = sum(x) > 0   （x の符号で両方の枝が選ばれる）
;;;;   then : a*b + a*v    （operands a=x, b=w。v は閉包で捕まえる）
;;;;   else : a*a*w2       （w2 = w を閉包で捕まえる。b は使わない）
;;;; 期待値は jvp / transpose を使わない f64 の中心差分と、内積（随伴性）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defmacro %cj-f (x w v)
  "with-tracing の本体に展開される（walker が見る式にするためマクロ）。"
  `(nb:cond* (> (nb:reduce-sum ,x :axes '(0)) 0.0)
            (nb:with-tracing (a b) (+ (* a b) (* a ,v)))
            (nb:with-tracing (a b) (* (* a a) ,w))
            ,x ,w))

(defun %cj-avals (&optional (dtype :f64))
  (list (nb:make-aval '(3) dtype) (nb:make-aval '(3) dtype) (nb:make-aval '(3) dtype)))

(defun %cj-graph (&optional (dtype :f64))
  (nb::trace-to-graph (nb:with-tracing (x w v) (%cj-f x w v)) (%cj-avals dtype)))

(defun %cj-zero-else-graph ()
  "else の出力が x に依存しない（x だけに接線を渡すと else の接線がゼロ）cond。"
  (nb::trace-to-graph
   (nb:with-tracing (x w)
     (nb:cond* (> (nb:reduce-sum x :axes '(0)) 0.0)
               (nb:with-tracing (a b) (* a b))
               (nb:with-tracing (a b) (+ b b))
               x w))
   (list (nb:make-aval '(3) :f64) (nb:make-aval '(3) :f64))))

(defun %cj-arrays (seed n)
  (loop for i below n
        collect (make-random-array (make-array-spec '(3) :f64) :seed (+ seed i))))

(defun %cj-tangents (seed n &optional zero-from)
  (loop for i below n
        collect (if (and zero-from (>= i zero-from))
                    (make-array '(3) :element-type 'double-float :initial-element 0d0)
                    (random-tangent (nb:make-aval '(3) :f64) :seed (+ seed 100 i)))))

(defun %cj-close-p (actual expected)
  (every (lambda (a e) (allclose a e :dtype :f64 :rtol *autodiff-rtol* :atol *autodiff-atol*))
         actual expected))

(defun %cj-jvp-tangents (graph primals tangents &key (nonzero nil nonzero-p))
  (let* ((jvp (if nonzero-p (nb::jvp-graph graph :nonzero nonzero) (nb::jvp-graph graph)))
         (inputs (append primals (if nonzero-p
                                     (loop for tg in tangents for f in nonzero when f collect tg)
                                     tangents))))
    (nthcdr (length (nb:graph-outvars graph))
            (multiple-value-list (apply #'nb:eval-graph jvp inputs)))))

(test cond-jvp/matches-central-difference
  "cond の jvp は f64 の中心差分と一致する（両方の枝が選ばれる入力。captured の v・w の接線を含む）。"
  (let ((graph (%cj-graph)))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let ((primals (%cj-arrays seed 3))
                          (tangents (%cj-tangents seed 3)))
                      (%cj-close-p (%cj-jvp-tangents graph primals tangents)
                                   (central-difference-jvp graph primals tangents))))
                  :regression-id cond-jvp/central-difference
                  :regression-file (regression-path "cond-jvp-central-difference")))))

(test cond-jvp/both-branches-are-taken
  "上の PBT が両方の枝を通ること（sum(x) の符号が両方出る seed がある）の確認。"
  (let ((signs (loop for seed below 40
                     collect (plusp (reduce #'+ (let ((x (first (%cj-arrays seed 3))))
                                                  (loop for i below 3 collect (row-major-aref x i))))))))
    (is (member t signs))
    (is (member nil signs))))

(test cond-jvp/zero-tangent-branch-is-instantiated
  "片方の枝だけで接線が非ゼロ（x だけに接線を渡すと else の出力は x に依存しない）でも、
両枝の接線の aval が揃い、値が中心差分と一致する。"
  (let ((graph (%cj-zero-else-graph)))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let ((primals (%cj-arrays seed 2))
                          (tangents (%cj-tangents seed 2 1)))
                      (%cj-close-p (%cj-jvp-tangents graph primals tangents :nonzero '(t nil))
                                   (central-difference-jvp graph primals tangents))))
                  :regression-id cond-jvp/zero-branch
                  :regression-file (regression-path "cond-jvp-zero-branch")))))

(test cond-jvp/is-linear-in-tangents
  "jvp(a t1 + b t2) = a jvp(t1) + b jvp(t2)。"
  (let ((graph (%cj-graph)))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((primals (%cj-arrays seed 3))
                           (t1 (%cj-tangents seed 3))
                           (t2 (%cj-tangents (+ seed 500) 3))
                           (a (+ 0.5d0 (mod seed 7)))
                           (b (- (mod seed 5) 2.5d0))
                           (combine (lambda (u v) (%sum-array (%scale-array u a) (%scale-array v b)))))
                      (%cj-close-p (%cj-jvp-tangents graph primals (mapcar combine t1 t2))
                                   (mapcar combine
                                           (%cj-jvp-tangents graph primals t1)
                                           (%cj-jvp-tangents graph primals t2)))))
                  :regression-id cond-jvp/linear
                  :regression-file (regression-path "cond-jvp-linear")))))

(test cond-grad/matches-central-difference
  "grad (cond* を含む損失) は、f64 の中心差分の勾配と一致する（両方の枝が選ばれる入力）。"
  (let* ((f (nb:with-tracing (x w v) (nb:reduce-sum (%cj-f x w v) :axes '(0))))
         (loss (nb::trace-to-graph f (%cj-avals)))
         (grad (nb:grad f :argnums '(0 1 2))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let ((primals (%cj-arrays seed 3)))
                      (%cj-close-p (apply grad primals)
                                   (central-difference-gradient loss primals))))
                  :regression-id cond-grad/central-difference
                  :regression-file (regression-path "cond-grad-central-difference")))))

(test cond-transpose/satisfies-adjoint-identity
  "linearize した cond の線形部分を transpose すると、内積 <ct, L t> = <L^T ct, t> が成り立つ
（線形部分は線形な cond を含む）。"
  (let* ((graph (%cj-graph))
         (lin (nb::linearize-graph graph))
         (n-res (nb::linearization-n-residuals lin))
         (primal (nb::linearization-primal-graph lin))
         (linear (nb::linearization-linear-graph lin))
         (transposed (nb::transpose-graph linear n-res)))
    (is (find :cond (nb:graph-eqns linear)
              :key (lambda (e) (nb::primitive-name (nb:eqn-prim e)))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((primals (%cj-arrays seed 3))
                           (residuals (nthcdr 1 (multiple-value-list (apply #'nb:eval-graph primal primals))))
                           (tangents (%cj-tangents seed 3))
                           (ct (random-cotangent (nb:make-aval '(3) :f64) :seed (+ seed 900)))
                           (forward (first (multiple-value-list
                                            (apply #'nb:eval-graph linear (append residuals tangents)))))
                           (backward (multiple-value-list
                                      (apply #'nb:eval-graph transposed (append residuals (list ct))))))
                      (let ((lhs (inner-product ct forward))
                            (rhs (reduce #'+ (mapcar #'inner-product backward tangents))))
                        (<= (abs (- lhs rhs)) (+ 1d-9 (* 1d-9 (abs lhs)))))))
                  :regression-id cond-transpose/adjoint
                  :regression-file (regression-path "cond-transpose-adjoint")))))

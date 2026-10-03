;;;; while-loop の jvp ルール（issue #134）。
;;;;
;;;; 対象は limit・w・x を引数に取る f64 の while-loop:
;;;;   carry = (i, x, y)   i はカウンタ（rank 0）、x は (3)、y は rank 0 で初期値が定数 0
;;;;   cond  : i < limit   （limit は cond が閉包で捕まえる）
;;;;   body  : i+1, x*0.5 + w*0.1, y*0.9 + sum(x*w)   （w は body が閉包で捕まえる）
;;;; y の初期値は定数なので接線がゼロから始まり、本体を1回通ると x の接線で非ゼロになる
;;;; （JAX の不動点の回帰）。期待値は jvp-graph を使わない f64 の中心差分。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %wlj-graph ()
  (nb::trace-to-graph
   (nb:with-tracing (limit w x)
     (let ((result (nb:while-loop
                    (nb:with-tracing (c) (< (first c) limit))
                    (nb:with-tracing (c)
                      (list (+ (first c) 1.0)
                            (+ (* (second c) 0.5) (* w 0.1))
                            (+ (* (third c) 0.9) (nb:reduce-sum (* (second c) w) :axes '(0)))))
                    (list (nb::%scalar-array 0d0 :f64) x (nb::%scalar-array 0d0 :f64)))))
       (values (third result) (second result))))
   (list (nb:make-aval '() :f64) (nb:make-aval '(3) :f64) (nb:make-aval '(3) :f64))))

(defun %wlj-primals (seed)
  (list (nb::%scalar-array (float (mod seed 6) 1d0) :f64)
        (make-random-array (make-array-spec '(3) :f64) :seed seed)
        (make-random-array (make-array-spec '(3) :f64) :seed (+ seed 1))))

(defun %wlj-tangents (seed &key (zero-limit t) (zero-w nil) (zero-x nil))
  (flet ((tangent (shape s zero)
           (if zero
               (make-array shape :element-type 'double-float :initial-element 0d0)
               (random-tangent (nb:make-aval shape :f64) :seed s))))
    (list (tangent '() (+ seed 10) zero-limit)
          (tangent '(3) (+ seed 11) zero-w)
          (tangent '(3) (+ seed 12) zero-x))))

(defun %wlj-jvp-tangent-outputs (graph primals tangents &key (nonzero nil nonzero-p))
  "graph を jvp 変換して評価し、接線の出力（後半）のリストを返す。NONZERO を渡すと、
その入力にだけ接線を渡す（TANGENTS は全入力の分。非ゼロの入力のものだけ使う）。"
  (let* ((jvp (if nonzero-p (nb::jvp-graph graph :nonzero nonzero) (nb::jvp-graph graph)))
         (inputs (append primals (if nonzero-p
                                     (loop for tg in tangents for flag in nonzero when flag collect tg)
                                     tangents)))
         (outs (multiple-value-list (apply #'nb:eval-graph jvp inputs))))
    (subseq outs (length (nb:graph-outvars graph)))))

(defun %wlj-close-p (actual expected)
  (every (lambda (a e) (allclose a e :dtype :f64 :rtol *autodiff-rtol* :atol *autodiff-atol*))
         actual expected))

(test while-loop-jvp/matches-central-difference
  "while-loop の jvp は、f64 の中心差分と一致する（反復回数 0〜5、captured の w の
接線と、初期値がゼロの carry y の接線を含む）。"
  (let ((graph (%wlj-graph)))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let ((primals (%wlj-primals seed))
                          (tangents (%wlj-tangents seed)))
                      (%wlj-close-p (%wlj-jvp-tangent-outputs graph primals tangents)
                                    (central-difference-jvp graph primals tangents))))
                  :regression-id while-loop-jvp/central-difference
                  :regression-file (regression-path "while-loop-jvp-central-difference")))))

(test while-loop-jvp/is-linear-in-tangents
  "jvp(a t1 + b t2) = a jvp(t1) + b jvp(t2)（同じ primal での接線について線形）。"
  (let ((graph (%wlj-graph)))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((primals (%wlj-primals seed))
                           (t1 (%wlj-tangents seed))
                           (t2 (%wlj-tangents (+ seed 500)))
                           (a (+ 0.5d0 (mod seed 7)))
                           (b (- (mod seed 5) 2.5d0))
                           (combined (mapcar (lambda (u v) (%sum-array (%scale-array u a) (%scale-array v b)))
                                             t1 t2))
                           (actual (%wlj-jvp-tangent-outputs graph primals combined))
                           (expected (mapcar (lambda (u v) (%sum-array (%scale-array u a) (%scale-array v b)))
                                             (%wlj-jvp-tangent-outputs graph primals t1)
                                             (%wlj-jvp-tangent-outputs graph primals t2))))
                      (%wlj-close-p actual expected)))
                  :regression-id while-loop-jvp/linear
                  :regression-file (regression-path "while-loop-jvp-linear")))))

(defun %wlj-while-eqn (graph)
  (find :while-loop (nb:graph-eqns graph)
        :key (lambda (e) (nb::primitive-name (nb:eqn-prim e)))))

(test while-loop-jvp/initially-zero-carry-tangent-becomes-nonzero
  "x だけに接線を渡す（limit・w の接線はゼロ、y の初期値は定数）。y の接線は最初ゼロだが
本体で x の接線が流れ込むので、不動点で y も接線の carry になる: carry は i・x・y に
x と y の接線を足した 3 + 2 = 5 個、オペランドは carry 5 + limit・w の 2 = 7 個。値も
中心差分と一致する。"
  (let* ((graph (%wlj-graph))
         (nonzero '(nil nil t))
         (jvp (nb::jvp-graph graph :nonzero nonzero))
         (eqn (%wlj-while-eqn jvp)))
    (is (not (null eqn)))
    (is (= 5 (getf (nb:eqn-params eqn) :n-carries)))
    (is (= 7 (length (nb:eqn-invars eqn))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((primals (%wlj-primals seed))
                           (tangents (%wlj-tangents seed :zero-w t)))
                      (%wlj-close-p (%wlj-jvp-tangent-outputs graph primals tangents :nonzero nonzero)
                                    (central-difference-jvp graph primals tangents))))
                  :regression-id while-loop-jvp/initially-zero
                  :regression-file (regression-path "while-loop-jvp-initially-zero")))))

(test while-loop-jvp/no-tangent-carries-for-zero-tangents
  "接線が全部ゼロなら、jvp 変換した graph の while-loop は元と同じ carry（3 個）のまま。"
  (let* ((jvp (nb::jvp-graph (%wlj-graph) :nonzero '(nil nil nil)))
         (eqn (%wlj-while-eqn jvp)))
    (is (= 3 (getf (nb:eqn-params eqn) :n-carries)))
    (is (= 5 (length (nb:eqn-invars eqn))))))

(test while-loop-jvp/grad-through-while-loop-signals-autodiff-error
  "逆モードは対応しない: grad が while-loop を通ると、プリミティブ名 :while-loop を持つ
autodiff-error になる。"
  (let ((f (nb:with-tracing (w x)
             (nb:reduce-sum
              (first (nb:while-loop
                      (nb:with-tracing (c) (< (second c) 3.0))
                      (nb:with-tracing (c) (list (* (first c) w) (+ (second c) 1.0)))
                      (list x (nb::%scalar-array 0.0 :f32))))
              :axes '(0)))))
    (handler-case
        (progn
          (funcall (nb:grad f)
                   (make-random-array (make-array-spec '(3) :f32) :seed 1)
                   (make-random-array (make-array-spec '(3) :f32) :seed 2))
          (fail "autodiff-error にならなかった"))
      (nb::autodiff-error (c)
        (is (search ":WHILE-LOOP" (princ-to-string c)))))))

(test while-loop-jvp/grad-ignores-an-unused-while-loop
  "結果に効かない while-loop は、grad の途中の DCE で消えるので、逆モードに対応していなくても
エラーにならない（JAX と同じ）。勾配は while-loop が無い関数のものと一致する。"
  (let* ((f (nb:with-tracing (x)
              (let ((unused (nb:while-loop
                             (nb:with-tracing (c) (< (second c) 3.0))
                             (nb:with-tracing (c) (list (* (first c) 2.0) (+ (second c) 1.0)))
                             (list x (nb::%scalar-array 0.0 :f32)))))
                (declare (ignorable unused))
                (nb:reduce-sum (* x x) :axes '(0)))))
         (x (make-random-array (make-array-spec '(3) :f32) :seed 4)))
    (is (allclose (funcall (nb:grad f) x) (nb::%t-mul x (nb::%scalar-array 2.0 :f32)) :dtype :f32))))

(test while-loop-jvp/grad-through-a-used-while-loop-still-errors-after-dce
  "結果に効く while-loop を通した grad は、DCE の後でも autodiff-error（メッセージは
プリミティブ名 :WHILE-LOOP を含み、while 専用の文言ではなく一般的な説明）。"
  (let ((f (nb:with-tracing (x)
             (nb:reduce-sum
              (first (nb:while-loop
                      (nb:with-tracing (c) (< (second c) 3.0))
                      (nb:with-tracing (c) (list (* (first c) 2.0) (+ (second c) 1.0)))
                      (list x (nb::%scalar-array 0.0 :f32))))
              :axes '(0)))))
    (signals nb::autodiff-error
      (funcall (nb:grad f) (make-random-array (make-array-spec '(3) :f32) :seed 4)))))

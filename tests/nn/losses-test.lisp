;;;; soft-target-cross-entropy（Softmax Cancellation の修正、small）。
;;;;
;;;; 蒸留の交差エントロピー -sum(softmax(z_t) log-softmax(z_s)) の勾配 q - p を素朴に自動微分すると、
;;;; z_s が z_t に近いとき桁を失い、z_s = z_t でも 0 にならない（Barron ら、docs/numerics.md）。
;;;; 守らせる性質:
;;;; - z_s = z_t なら勾配はちょうど 0（全要素、f32 / f64 / bf16）。
;;;; - z_s が z_t のすぐ近く（差 10^-7〜10^-1）でも勾配が double の参照 q - p と一致し、和が 0。
;;;; - 論文の例（z_t = [2, 1, 0]、z_s = z_t + [-1e-7, 1e-7, 0]）で厳密な勾配と一致する。
;;;; - 値は -sum(p log q)、生徒についての2階微分は diag(q) - q q^T、教師には勾配が流れない。
;;;; - -inf のロジットで NaN を出さない、形の違いは error。

(in-package #:nabla.nn.tests)

(in-suite :nabla.small)

(defun %soft-ce ()
  (nabla:with-tracing (zs zt) (soft-target-cross-entropy zs zt)))

(defun %reference-soft-ce-grad (zs zt)
  "q - p を double で（q, p は double の log-softmax の参照から）。"
  (map 'vector #'- (%reference-softmax zs) (%reference-softmax zt)))

(defun %perturbed (row seed scale)
  "ROW の各要素に [-SCALE, SCALE) の乱数を足した f32 の行。"
  (let ((state (sb-ext:seed-random-state seed))
        (result (make-array (length row) :element-type 'single-float)))
    (dotimes (i (length row) result)
      (setf (aref result i) (+ (aref row i) (coerce (- (random (* 2d0 scale) state) scale) 'single-float))))))

(test losses/matches-the-paper-example
  "z_t = [2, 1, 0]、z_s = z_t + [-1e-7, 1e-7, 0]（f32）の勾配は厳密な値 [-4.596e-8, 4.144e-8, 4.513e-9]
と相対誤差 1e-3 で一致する（論文の 3 節。素朴な実装は [-5.960e-8, 1.452e-8, -1.123e-9]）。"
  (let* ((zt (%f32-vector '(2 1 0)))
         (zs (let ((v (copy-seq zt))) (decf (aref v 0) 1e-7) (incf (aref v 1) 1e-7) v))
         (g (identity (funcall (nabla:grad (%soft-ce)) zs zt))))
    (is (%close-relative-p g #(-4.596d-8 4.144d-8 4.513d-9) 1d-3) "勾配 ~S" g)))

(def-layer-property losses/gradient-is-exactly-zero-when-student-equals-teacher
  "z_s = z_t なら soft-target-cross-entropy の z_s についての勾配は全要素がちょうど 0（f32、余裕の
大きい行を含む）。"
  (seed n margin)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 8) (uniform-integer :lo 0 :hi 40))
  (let* ((z (%dominated-row seed n 0 margin))
         (g (identity (funcall (nabla:grad (%soft-ce)) z z))))
    (every #'zerop g)))

(test losses/gradient-is-exactly-zero-at-equality-for-f64-and-bf16
  "f64 と bf16（トレースしたグラフを eager で評価）でも z_s = z_t の勾配はちょうど 0。"
  (let ((z64 (make-array 4 :element-type 'double-float :initial-contents '(1.5d0 -2d0 0.25d0 3d0))))
    (is (every #'zerop (identity (funcall (nabla:grad (%soft-ce)) z64 z64)))))
  (let* ((bits (nabla:convert (%f32-vector '(1.5 -2 0.25 3)) :bf16))
         (aval (nabla:make-aval '(4) :bf16))
         (graph (nabla:trace-to-graph (nabla:grad (%soft-ce)) (list aval aval)))
         (g (nabla:eval-graph graph bits bits)))
    (is (every #'zerop (nabla::decode-float16-array (if (listp g) (first g) g) :bf16)))))

(def-layer-property losses/gradient-is-accurate-near-the-teacher
  "z_s = z_t + 差（各要素 [-10^-e, 10^-e)、e は 1〜7）のとき、勾配が double の参照 q - p と一致する
（成分ごとに相対 1e-3、床は最大の成分の 1e-3。素朴な実装は差 1e-7 で符号まで変わる）。和は 0。"
  (seed n e)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 8) (uniform-integer :lo 1 :hi 7))
  (let* ((zt (%random-row seed n))
         (zs (%perturbed zt (1+ seed) (expt 10d0 (- e))))
         (g (identity (funcall (nabla:grad (%soft-ce)) zs zt)))
         (expected (%reference-soft-ce-grad zs zt))
         (scale (reduce #'max expected :key #'abs)))
    (and (every (lambda (a x) (<= (abs (- a x)) (+ (* 1d-3 (abs x)) (* 1d-3 scale)))) g expected)
         (%gradient-sum-small-p g))))

(def-layer-property losses/gradient-is-accurate-for-a-shifted-student
  "z_s = z_t + C + 差（C は 1〜50 の定数、差は各要素 [-1e-9, 1e-9)、f64）でも、勾配が double の参照
q - p と一致する（softmax は定数のずらしに不変なので q ≈ p。成分ごとに相対 1e-3、床は最大の成分の
1e-3）。f32 では z_t + C の丸め（50 で ulp 3.8e-6）が差を消すので f64 で見る。中心化
d = delta - sum(p delta) を外す変異はこの性質では区別できない（中心化は s の値を変えず、精度への
効果も差 z_s - z_t の丸めと同じ大きさ。生き残る変異体として記録している）。"
  (seed n c)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 8) (uniform-integer :lo 1 :hi 50))
  (let* ((state (sb-ext:seed-random-state seed))
         (zt (make-array n :element-type 'double-float))
         (zs (make-array n :element-type 'double-float)))
    (dotimes (i n)
      (setf (aref zt i) (- (random 6d0 state) 3d0)
            (aref zs i) (+ (aref zt i) c (- (random 2d-9 state) 1d-9))))
    (let* ((g (funcall (nabla:grad (%soft-ce)) zs zt))
           (expected (%reference-soft-ce-grad zs zt))
           (scale (reduce #'max expected :key #'abs)))
      (every (lambda (a x) (<= (abs (- a x)) (+ (* 1d-3 (abs x)) (* 1d-3 scale)))) g expected))))

(def-layer-property losses/gradient-is-exactly-zero-for-a-shifted-teacher
  "z_s = z_t + C（C は整数、z_t は 1/64 の倍数なので和は厳密）なら softmax(z_s) = softmax(z_t) で、勾配は
全要素がちょうど 0（中心化を sum(p delta) で取ると sum(p) = 1 の丸めで eps * C の勾配が残る）。"
  (seed n c)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 8) (uniform-integer :lo -100 :hi 100))
  (let* ((state (sb-ext:seed-random-state seed))
         (zt (make-array n :element-type 'single-float))
         (zs (make-array n :element-type 'single-float)))
    (dotimes (i n)
      (setf (aref zt i) (/ (- (random 384 state) 192) 64.0)
            (aref zs i) (+ (aref zt i) c)))
    (every #'zerop (funcall (nabla:grad (%soft-ce)) zs zt))))

(def-layer-property losses/value-is-the-cross-entropy
  "値は -sum(softmax(z_t) log-softmax(z_s))（double の参照と相対誤差 1e-5）。2 次元の入力は行ごと。"
  (seed n)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 8))
  (let* ((zs (%random-row seed n))
         (zt (%random-row (+ 3 seed) n))
         (expected (- (reduce #'+ (map 'vector #'* (%reference-softmax zt) (%reference-log-softmax zs)))))
         (value (aref (funcall (%soft-ce) zs zt)))
         (two (make-array (list 2 n) :element-type 'single-float))
         (two-t (make-array (list 2 n) :element-type 'single-float)))
    (dotimes (i n)
      (setf (aref two 0 i) (aref zs i) (aref two 1 i) (aref zs i)
            (aref two-t 0 i) (aref zt i) (aref two-t 1 i) (aref zt i)))
    (let ((rows (funcall (%soft-ce) two two-t)))
      (and (<= (abs (- value expected)) (* 1d-5 (abs expected)))
           (equal '(2) (array-dimensions rows))
           (= (aref rows 0) (aref rows 1) value)))))

(def-layer-property losses/student-hessian-is-the-softmax-jacobian
  "生徒についてのヘッセ行列とベクトルの積（grad の grad）は (diag(q) - q q^T) v（double の参照と
相対 1e-3、床は最大の成分の 1e-6）。教師についての勾配は 0。"
  (seed n)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 6))
  (let* ((zs (%random-row seed n))
         (zt (%random-row (+ 5 seed) n))
         (v (%random-row (+ 9 seed) n))
         (grad (nabla:grad (%soft-ce)))
         (hv (identity (funcall (nabla:grad (nabla:with-tracing (zs zt)
                                           (nabla:reduce-sum (* (funcall grad zs zt) v) :axes '(0))))
                             zs zt)))
         (expected (%reference-softmax-vjp zs v))
         (scale (reduce #'max expected :key #'abs))
         (teacher-grad (second (funcall (nabla:grad (%soft-ce) :argnums '(0 1)) zs zt))))
    (and (every (lambda (a x) (<= (abs (- a x)) (+ (* 1d-3 (abs x)) (* 1d-6 scale)))) hv expected)
         (every #'zerop teacher-grad))))

(test losses/infinities-and-shape-errors
  "教師が -inf の位置は目標の確率 0（値・勾配とも有限）、生徒も -inf ならその位置の勾配は 0。生徒だけが
-inf なら損失は +inf、その位置の勾配は 0、他の位置は q - p。形が違えば error。"
  (let* ((-inf sb-ext:single-float-negative-infinity)
         (zs (%f32-vector '(1 2 3)))
         (zt (%f32-vector (list 0 -inf 1)))
         (g (identity (funcall (nabla:grad (%soft-ce)) zs zt))))
    (is (notany #'sb-ext:float-nan-p g))
    (is (%close-relative-p g (%reference-soft-ce-grad zs (vector 0 -1000 1)) 1d-5) "~S" g)
    ;; 値は教師の確率が正の位置だけの -sum(p log q)
    (let ((expected (- (reduce #'+ (map 'vector (lambda (pj lq) (if (zerop pj) 0 (* pj lq)))
                                         (%reference-softmax (vector 0 -1000 1)) (%reference-log-softmax zs))))))
      (is (<= (abs (- (aref (funcall (%soft-ce) zs zt)) expected)) (* 1d-6 (abs expected)))))
    (let* ((zs (%f32-vector (list 1 -inf 3)))
           (value (aref (funcall (%soft-ce) zs zt)))
           (g (identity (funcall (nabla:grad (%soft-ce)) zs zt))))
      (is (not (sb-ext:float-nan-p value)))
      (is (notany #'sb-ext:float-nan-p g))
      (is (zerop (aref g 1)))))
  (let* ((zs (%f32-vector (list 1 sb-ext:single-float-negative-infinity 3)))
         (zt (%f32-vector '(0 0 1)))
         (g (funcall (nabla:grad (%soft-ce)) zs zt))
         (expected (%reference-soft-ce-grad (vector 1 -1000 3) zt)))
    ;; 生徒だけが -inf: 損失は +inf、その位置の勾配は 0、他の位置は q - p（相対 1e-5）
    (is (= sb-ext:single-float-positive-infinity (aref (funcall (%soft-ce) zs zt))))
    (is (zerop (aref g 1)) "~S" g)
    (is (%close-relative-p (vector (aref g 0) (aref g 2)) (vector (aref expected 0) (aref expected 2)) 1d-5) "~S" g))
  (let ((z (%f32-vector (list 1 sb-ext:single-float-negative-infinity 3))))
    (is (every #'zerop (funcall (nabla:grad (%soft-ce)) z z)) "-inf を含む z_s = z_t"))
  (signals error (soft-target-cross-entropy (%f32-vector '(1 2)) (%f32-vector '(1 2 3)))))

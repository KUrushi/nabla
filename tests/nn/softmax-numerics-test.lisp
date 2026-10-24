;;;; softmax / log-softmax の数値の正しさ（Softmax Collapse の修正、small）。
;;;;
;;;; Barron ら「A Fix for Softmax and Cross-Entropy」（docs/numerics.md）の Softmax Collapse:
;;;; 1 つのロジットが他より ln(1/eps)（bf16 で約 5.5、f32 で約 16.6、f64 で約 36.7）以上大きいと、
;;;; 素朴な log-softmax は 1 + s を 1 に丸め、値と最大値の位置の勾配 p - 1 を 0 にする（他の勾配は
;;;; 0 にならないので、勾配の和が 0 でなくなる）。守らせる性質:
;;;; - 論文の例（z = [9, -9, -9]、f32）の値と勾配がビット単位で論文と同じ。
;;;; - 余裕（最大値と他の差）が 0〜80 のどこでも、値と交差エントロピーの勾配が double の参照
;;;;   （最大値を除いた和を log1p する、実装とは別のループ）と相対誤差で一致し、勾配の和が 0。
;;;; - 余接線が任意のとき、log-softmax / softmax の VJP が参照と一致し、log-softmax の VJP の和が 0。
;;;; - 2階微分（交差エントロピーのヘッセ行列とベクトルの積）も崩れない。
;;;; - 同点の最大値・置換・-inf・NaN・長さ 1 の軸・bf16 での振る舞い。

(in-package #:nabla.nn.tests)

(in-suite :nabla.small)

;;; --- 参照（double） ---

(defun %row-max-index (row)
  (let ((k 0))
    (loop for i from 1 below (length row) when (> (aref row i) (aref row k)) do (setf k i))
    k))

(defun %reference-log-softmax (row)
  "ROW（実数のベクタ）の log-softmax を double で返す。最大値（先頭）を除いた exp の和 s を
log1p するので、余裕が大きくても桁を失わない（素朴な log(1 + s) は f64 でも 36.7 を超えると 0）。"
  (let* ((x (map 'vector (lambda (v) (coerce v 'double-float)) row))
         (k (%row-max-index x))
         (m (aref x k))
         (s (loop for j below (length x) unless (= j k) sum (exp (- (aref x j) m)))))
    (map 'vector (lambda (v) (- (- v m) (sb-kernel:%log1p s))) x)))

(defun %reference-softmax (row)
  (map 'vector #'exp (%reference-log-softmax row)))

(defun %reference-log-softmax-vjp (row ct)
  "log-softmax の VJP: ct_i - p_i sum(ct)。最大値の位置は ct_k sum_{j/=k} p_j - p_k sum_{j/=k} ct_j
（差を取らずに直接）。"
  (let* ((p (%reference-softmax row))
         (k (%row-max-index (map 'vector (lambda (v) (coerce v 'double-float)) row)))
         (ct (map 'vector (lambda (v) (coerce v 'double-float)) ct))
         (total (reduce #'+ ct)))
    (let ((result (map 'vector (lambda (c pj) (- c (* pj total))) ct p)))
      (setf (aref result k)
            (- (* (aref ct k) (loop for j below (length p) unless (= j k) sum (aref p j)))
               (* (aref p k) (loop for j below (length p) unless (= j k) sum (aref ct j)))))
      result)))

(defun %reference-softmax-vjp (row ct)
  "softmax の VJP p_i (ct_i - sum_j p_j ct_j)。最大値の位置は p_k sum_{j/=k} p_j (ct_k - ct_j) で
直接計算する（差を取ると桁を失う）。"
  (let* ((p (%reference-softmax row))
         (k (%row-max-index (map 'vector (lambda (v) (coerce v 'double-float)) row)))
         (ct (map 'vector (lambda (v) (coerce v 'double-float)) ct))
         (dot (loop for i below (length p) sum (* (aref p i) (aref ct i)))))
    (let ((result (map 'vector (lambda (pj c) (* pj (- c dot))) p ct)))
      (setf (aref result k)
            (* (aref p k) (loop for j below (length p) unless (= j k) sum (* (aref p j) (- (aref ct k) (aref ct j))))))
      result)))

(defun %close-relative-p (actual expected rtol)
  "各要素が |a - e| <= rtol |e|（e = 0 なら a = 0）。勾配が 0 に丸まる不具合を見逃さないよう、
絶対誤差の床は置かない（f32 の非正規数の範囲には入れない入力で使う）。"
  (every (lambda (a e) (<= (abs (- a e)) (* rtol (abs e)))) actual expected))

(defun %f32-vector (values)
  (make-array (length values) :element-type 'single-float
                              :initial-contents (map 'list (lambda (v) (coerce v 'single-float)) values)))

(defun %one-hot (k n)
  (let ((v (make-array n :element-type 'single-float :initial-element 0.0)))
    (setf (aref v k) 1.0)
    v))

(defun %cross-entropy (onehot)
  "交差エントロピー -sum(ONEHOT * log-softmax(x))（ONEHOT は配列の定数）のトレースできる関数。"
  (nabla:with-tracing (x) (- (nabla:reduce-sum (* (log-softmax x) onehot) :axes '(0)))))

(defun %vjp (fn x ct)
  "FN（1引数・配列を返す）の X での余接線 CT の VJP。"
  (funcall (nabla:grad (nabla:with-tracing (x) (nabla:reduce-sum (* (funcall fn x) ct) :axes '(0)))) x))

(defun %dominated-row (seed n k margin)
  "長さ N、位置 K の値が他の最大値より MARGIN だけ大きい f32 の行（他は [-3, 3)）。"
  (let ((state (sb-ext:seed-random-state seed))
        (row (make-array n :element-type 'single-float)))
    (dotimes (i n) (setf (aref row i) (- (random 6.0 state) 3.0)))
    (setf (aref row k) 0.0)
    (setf (aref row k) (+ (reduce #'max row :key (lambda (v) v) :initial-value -3.0) margin))
    row))

(defun %random-row (seed n)
  (let ((state (sb-ext:seed-random-state seed))
        (row (make-array n :element-type 'single-float)))
    (dotimes (i n row) (setf (aref row i) (- (random 6.0 state) 3.0)))))

(defun %gradient-sum-small-p (g &optional (eps 6d-8))
  "勾配の和が、成分の絶対値の和に対して丸め誤差の範囲で 0（|sum g| <= 4 n eps sum |g|）。"
  (let ((sum (reduce #'+ g :key (lambda (v) (coerce v 'double-float))))
        (abs-sum (reduce #'+ g :key (lambda (v) (abs (coerce v 'double-float))))))
    (<= (abs sum) (* 4 (length g) eps abs-sum))))

;;; --- 論文の例 ---

(test softmax-numerics/matches-the-paper-example
  "z = [9, -9, -9]（f32）、正解 0 のとき、log-softmax は [-3.046e-8, -18, -18]、交差エントロピーは
3.046e-8、その勾配は [-3.046e-8, 1.523e-8, 1.523e-8]、softmax(z)[0] の勾配は
[3.046e-8, -1.523e-8, -1.523e-8]（論文の 2 節。素朴な実装ではどれも 0 が混ざる）。"
  (let* ((z (%f32-vector '(9 -9 -9)))
         (ce (%cross-entropy (%one-hot 0 3)))
         (ls (log-softmax z))
         (g (funcall (nabla:grad ce) z))
         (gs (%vjp #'softmax z (%one-hot 0 3))))
    (is (%close-relative-p ls #(-3.046d-8 -18d0 -18d0) 1d-3) "log-softmax ~S" ls)
    (is (%close-relative-p (list (aref (funcall ce z))) '(3.046d-8) 1d-3))
    (is (%close-relative-p g #(-3.046d-8 1.523d-8 1.523d-8) 1d-3) "勾配 ~S" g)
    (is (%close-relative-p gs #(3.046d-8 -1.523d-8 -1.523d-8) 1d-3) "softmax の勾配 ~S" gs)))

;;; --- 性質 ---

(def-layer-property softmax-numerics/collapse-free-values-and-gradients
  "最大値が他より MARGIN（0〜80）大きい f32 の行で、log-softmax の各要素と、正解が最大値の位置の
交差エントロピーの勾配が、double の参照と相対誤差 1e-4 で一致する（最大値の位置の勾配も 0 に
丸まらない）。勾配の和は 0（丸め誤差の範囲）。"
  (seed n k-raw margin)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 8)
         (uniform-integer :lo 0 :hi 7) (uniform-integer :lo 0 :hi 80))
  (let* ((k (mod k-raw n))
         (x (%dominated-row seed n k margin))
         (g (funcall (nabla:grad (%cross-entropy (%one-hot k n))) x))
         (expected-g (%reference-log-softmax-vjp x (map 'vector #'- (%one-hot k n)))))
    (and (%close-relative-p (log-softmax x) (%reference-log-softmax x) 1d-4)
         (%close-relative-p g expected-g 1d-4)
         (%gradient-sum-small-p g))))

(def-layer-property softmax-numerics/vjps-match-the-reference-for-any-cotangent
  "任意の余接線 CT について、log-softmax と softmax の VJP が double の参照と一致する（相対誤差 1e-3、
余裕が大きい行を含む）。log-softmax の VJP の和は 0（丸め誤差の範囲）。"
  (seed n k-raw margin)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 8)
         (uniform-integer :lo 0 :hi 7) (uniform-integer :lo 0 :hi 40))
  (let* ((k (mod k-raw n))
         (x (%dominated-row seed n k margin))
         (ct (%random-row (1+ seed) n))
         (ls-vjp (%vjp #'log-softmax x ct))
         (sm-vjp (%vjp #'softmax x ct))
         (expected-sm (%reference-softmax-vjp x ct))
         (scale (reduce #'max expected-sm :key #'abs)))
    (and (%close-relative-p ls-vjp (%reference-log-softmax-vjp x ct) 1d-3)
         (%gradient-sum-small-p ls-vjp)
         ;; softmax の VJP は成分ごとに大きさが違う（最大値以外は p_j に比例）。最大の成分の 1e-6 を
         ;; 床にする（最大値の位置が 0 に丸まる不具合は、その位置が最大の成分と同じ桁なので見逃さない）
         (every (lambda (a e) (<= (abs (- a e)) (+ (* 1d-3 (abs e)) (* 1d-6 scale)))) sm-vjp expected-sm))))

(def-layer-property softmax-numerics/hessian-vector-product-is-collapse-free
  "交差エントロピーのヘッセ行列 H = diag(p) - p p^T とベクトル v の積（grad の grad）が、余裕が大きい
行でも参照（最大値の位置は p_k sum_{j/=k} p_j (v_k - v_j) を直接）と相対誤差 1e-3 で一致する。"
  (seed n k-raw margin)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 6)
         (uniform-integer :lo 0 :hi 5) (uniform-integer :lo 0 :hi 40))
  (let* ((k (mod k-raw n))
         (x (%dominated-row seed n k margin))
         (v (%random-row (+ 7 seed) n))
         (grad-ce (nabla:grad (%cross-entropy (%one-hot k n))))
         (hv (funcall (nabla:grad (nabla:with-tracing (x) (nabla:reduce-sum (* (funcall grad-ce x) v) :axes '(0)))) x))
         (scale (reduce #'max (%reference-softmax-vjp x v) :key #'abs)))
    ;; H v = p ⊙ (v - <p, v>) は softmax の VJP（余接線 v）と同じ式
    (every (lambda (a e) (<= (abs (- a e)) (+ (* 1d-3 (abs e)) (* 1d-6 scale))))
           hv (%reference-softmax-vjp x v))))

(def-layer-property softmax-numerics/equivariant-under-permutation
  "log-softmax(x の置換) = log-softmax(x) の同じ置換（相対誤差 1e-6。同点の最大値を含む）。"
  (seed n ties)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 7) (uniform-integer :lo 1 :hi 3))
  (let* ((x (%random-row seed n))
         (top (reduce #'max x)))
    (dotimes (i (min ties n)) (setf (aref x i) top))
    (let* ((perm (reverse (loop for i below n collect i)))
           (px (%f32-vector (mapcar (lambda (i) (aref x i)) perm)))
           (ls (log-softmax x))
           (pls (log-softmax px)))
      (loop for i below n for j in perm
            always (<= (abs (- (aref pls i) (aref ls j))) (* 1d-6 (max 1d0 (abs (aref ls j)))))))))

(test softmax-numerics/tied-maxima
  "最大値が c 個同点のとき、その位置の log-softmax は -log(c + tau)（tau は他の exp の和）で、正解が
同点の外にある交差エントロピーの勾配は同点の位置どうしで等しい。"
  (dolist (row '((3 3 0) (5 5 5 -1) (2 2 -30 -30) (0 0 -2)))
    (let* ((x (%f32-vector row))
           (n (length row))
           (ls (log-softmax x))
           (expected (%reference-log-softmax x))
           (g (funcall (nabla:grad (%cross-entropy (%one-hot (1- n) n))) x)))
      (is (%close-relative-p ls expected 1d-6) "~S: ~S" row ls)
      (is (= (aref g 0) (aref g 1)) "~S: 同点の勾配 ~S" row g)
      (is (%gradient-sum-small-p g) "~S: 勾配の和 ~S" row g))))

(test softmax-numerics/infinities-nan-and-length-one
  "-inf の要素は log-softmax が -inf・softmax が 0 で、他の要素と勾配に NaN を出さない。全部が -inf の
行と NaN を含む行は NaN（JAX と同じ）。長さ 1 の軸は値 0・勾配 0。"
  (let* ((-inf sb-ext:single-float-negative-infinity)
         (x (%f32-vector (list 1 -inf 0)))
         (ls (log-softmax x))
         (g (funcall (nabla:grad (%cross-entropy (%one-hot 0 3))) x)))
    (is (= -inf (aref ls 1)))
    (is (zerop (aref (softmax x) 1)))
    (is (%close-relative-p (vector (aref ls 0) (aref ls 2)) (vector (aref (%reference-log-softmax #(1 0)) 0)
                                                                    (aref (%reference-log-softmax #(1 0)) 1))
                           1d-6))
    (is (notany #'sb-ext:float-nan-p g) "勾配 ~S" g)
    (is (zerop (aref g 1)))
    (is (every #'sb-ext:float-nan-p (softmax (%f32-vector (list -inf -inf)))))
    (is (every #'sb-ext:float-nan-p
               (log-softmax (%f32-vector (list 1 (sb-kernel:make-single-float #x7fc00000) 0))))))
  (is (equalp #(0.0) (log-softmax (%f32-vector '(5)))))
  (is (equalp #(0.0) (funcall (nabla:grad (%cross-entropy (%one-hot 0 1))) (%f32-vector '(5))))))

(test softmax-numerics/other-axes-and-batches
  "axis 0 の softmax は転置して axis -1 で取ったものと等しく、行ごとの vmap は 2 次元の配列に
そのまま使ったものと等しい（余裕の大きい行を含む）。"
  (let* ((x (make-array '(2 3) :element-type 'single-float :initial-contents '((30.0 0.0 1.0) (1.0 2.0 -40.0))))
         (xt (make-array '(3 2) :element-type 'single-float)))
    (dotimes (i 2) (dotimes (j 3) (setf (aref xt j i) (aref x i j))))
    (let ((by-axis (log-softmax xt :axis 0))
          (by-row (log-softmax x))
          (by-vmap (funcall (nabla:vmap (nabla:with-tracing (r) (log-softmax r))) x)))
      (is (equalp by-row by-vmap))
      (dotimes (i 2) (dotimes (j 3) (is (= (aref by-row i j) (aref by-axis j i))))))))

(test softmax-numerics/bf16-keeps-the-target-gradient
  "bf16（精度の崖が約 5.5）でも、余裕 6〜30 の行の交差エントロピーの勾配は最大値の位置で 0 に丸まらず、
bf16 に丸めた入力の double の参照と bf16 の許容誤差（相対 2e-2）で一致する（jit してトレースした
グラフを eager で評価する）。"
  (dolist (margin '(6 10 20 30))
    (let* ((row (list margin 0 -1))
           (bits (nabla:convert (%f32-vector row) :bf16))
           (ce (nabla:with-tracing (x onehot) (- (nabla:reduce-sum (* (log-softmax x) onehot) :axes '(0)))))
           (graph (nabla:trace-to-graph (nabla:grad ce) (list (nabla:make-aval '(3) :bf16) (nabla:make-aval '(3) :bf16))))
           (g (nabla::decode-float16-array (nabla:eval-graph graph bits (nabla:convert (%one-hot 0 3) :bf16)) :bf16))
           (expected (%reference-log-softmax-vjp (nabla::decode-float16-array bits :bf16) #(-1 0 0))))
      (is (%close-relative-p g expected 2d-2) "余裕 ~D: ~S（期待 ~S）" margin g expected))))

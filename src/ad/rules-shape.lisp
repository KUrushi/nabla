;;;; ad/rules-shape: 形状演算・縮約・dot-general の jvp ルール（issue #81）と、それらの transpose ルール（#83、#84）。
;;;;
;;;; JAX の jax._src.lax.lax の reshape / broadcast_in_dim / transpose、
;;;; jax._src.lax.lax の reduce_sum / reduce_max（_reduce_chooser_jvp_rule）、
;;;; dot_general（_dot_general_jvp_lhs / _dot_general_jvp_rhs と同じ「積の微分」）に対応する。
;;;;
;;;; 制約（rules-elementwise.lisp 冒頭と同じ）: 接線は被演算子について線形な
;;;; プリミティブにしか流さない。ここでは reshape / broadcast-in-dim /
;;;; transpose / reduce-sum は接線に同じ演算を同じパラメタで適用するだけ、
;;;; dot-general は片側だけが接線の dot-general の和、reduce-max は接線に
;;;; 主値だけから作った係数（指示関数）を掛ける。いずれも接線は線形にしか
;;;; 使わない。

(in-package #:nabla)

(def-jvp-rule reshape (primals out tangents &key shape)
  (declare (ignore primals out))
  (%trace-eqn :reshape (list (first tangents)) :shape shape))

(def-jvp-rule broadcast-in-dim (primals out tangents &key shape dims)
  (declare (ignore primals out))
  (%trace-eqn :broadcast-in-dim (list (first tangents)) :shape shape :dims dims))

(def-jvp-rule transpose (primals out tangents &key perm)
  (declare (ignore primals out))
  (%trace-eqn :transpose (list (first tangents)) :perm perm))

(def-jvp-rule reduce-sum (primals out tangents &key axes)
  (declare (ignore primals out))
  (%trace-eqn :reduce-sum (list (first tangents)) :axes axes))

(def-jvp-rule reduce-max (primals out tangents &key axes)
  ;; JAX の _reduce_chooser_jvp_rule と同じ: 最大値を取る要素の指示関数 ind
  ;; （重複するときは reduce-sum(ind) で割って平均）。ind は主値だけから作る
  ;; ので、接線 t は reduce-sum(t * ind) / reduce-sum(ind) と線形にしか流れない。
  ;; :i1 は convert の入力にできないので select で 1 / 0 を作る。
  (let* ((x (first primals))
         (aval (tracer-aval x))
         (shape (aval-shape aval))
         (dtype (aval-dtype aval))
         (kept (loop for i below (length shape) unless (member i axes) collect i))
         (out-wide (%trace-eqn :broadcast-in-dim (list out) :shape shape :dims kept))
         (ind (%t-select (%t-compare x out-wide :eq)
                         (%lift-number-to 1 dtype shape)
                         (%lift-number-to 0 dtype shape))))
    (%t-div (%trace-eqn :reduce-sum (list (%t-mul (first tangents) ind)) :axes axes)
            (%trace-eqn :reduce-sum (list ind) :axes axes))))

(def-jvp-rule dot-general (primals out tangents &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
  ;; 積の微分: d(l · r) = dl · r + l · dr。ゼロの接線の項は作らない。
  (flet ((dot (lhs rhs)
           (%trace-eqn :dot-general (list lhs rhs)
                       :lhs-contracting lhs-contracting :rhs-contracting rhs-contracting
                       :lhs-batch lhs-batch :rhs-batch rhs-batch)))
    (destructuring-bind (lhs rhs) primals
      (destructuring-bind (lhs-tangent rhs-tangent) tangents
        (add-tangents
         (if (symbolic-zero-p lhs-tangent)
             (make-symbolic-zero (tracer-aval out))
             (dot lhs-tangent rhs))
         (if (symbolic-zero-p rhs-tangent)
             (make-symbolic-zero (tracer-aval out))
             (dot lhs rhs-tangent)))))))

;;; --- 以下は issue #83: 形状・縮約の transpose ルール。JAX の
;;; _reshape_transpose_rule / _transpose_transpose_rule /
;;; _broadcast_in_dim_transpose_rule / _reduce_sum_transpose_rule を写す。 ---

(def-transpose-rule reshape (ct invars &key shape)
  (declare (ignore shape))
  (list (%trace-eqn :reshape (list ct) :shape (aval-shape (undefined-primal-aval (first invars))))))

(def-transpose-rule transpose (ct invars &key perm)
  (declare (ignore invars))
  ;; 出力の軸 i は入力の軸 PERM[i]。逆置換 INV[PERM[i]] = i を適用する。
  (list (%trace-eqn :transpose (list ct)
                    :perm (loop for i below (length perm) collect (position i perm)))))

(def-transpose-rule broadcast-in-dim (ct invars &key shape dims)
  ;; 増えた軸（DIMS に無い出力の軸）と、サイズ1から広げた軸を足し、足した
  ;; サイズ1の軸を reshape で戻す。DIMS が昇順でなければ、足したあとの軸（出力の
  ;; 軸の昇順）をオペランドの軸の順に transpose で並べ替える。
  (let* ((in-shape (aval-shape (undefined-primal-aval (first invars))))
         (kept-operand-axes (loop for size in in-shape for d in dims for j from 0
                                  when (= size (nth d shape)) collect j))
         (kept-out-axes (sort (loop for j in kept-operand-axes collect (nth j dims)) #'<))
         (reduce-axes (loop for i below (length shape) unless (member i kept-out-axes) collect i))
         (summed (if reduce-axes (%trace-eqn :reduce-sum (list ct) :axes reduce-axes) ct))
         (perm (loop for j in kept-operand-axes collect (position (nth j dims) kept-out-axes)))
         (ordered (if (equal perm (loop for i below (length perm) collect i))
                      summed
                      (%trace-eqn :transpose (list summed) :perm perm))))
    (list (if (equal (aval-shape (tracer-aval ordered)) in-shape)
              ordered
              (%trace-eqn :reshape (list ordered) :shape in-shape)))))

(def-transpose-rule reduce-sum (ct invars &key axes)
  ;; 潰した軸を broadcast-in-dim で元の形に戻す。dims は残った軸。
  (let* ((shape (aval-shape (undefined-primal-aval (first invars))))
         (kept (loop for i below (length shape) unless (member i axes) collect i)))
    (list (%trace-eqn :broadcast-in-dim (list ct) :shape shape :dims kept))))

;;; --- 以下は issue #84: dot-general の transpose ルール。JAX の
;;; _dot_general_transpose_lhs / _dot_general_transpose_rhs を写す。 ---

(defun %dot-transpose-lhs (ct x-aval y x-contract y-contract x-batch y-batch &key swap-ans)
  "線形な X（aval X-AVAL）の余接線。Y は既知。CT と Y の dot-general（CT の Y 側の
自由次元と Y の自由次元を縮約し、バッチ次元を揃える）を、X の軸の順に transpose する。
SWAP-ANS が真のとき、出力の自由次元の並びが（Y の自由次元, X の自由次元）の順（右が線形のとき）。"
  (let* ((x-rank (length (aval-shape x-aval)))
         (y-rank (length (aval-shape (tracer-aval y))))
         (x-kept (loop for i below x-rank unless (or (member i x-contract) (member i x-batch)) collect i))
         (y-kept (loop for i below y-rank unless (or (member i y-contract) (member i y-batch)) collect i))
         (n-batch (length x-batch))
         (ans-batch (loop for i below n-batch collect i))
         (ans-y (loop for i below (length y-kept)
                      collect (+ n-batch i (if swap-ans 0 (length x-kept)))))
         ;; 結果の軸は バッチ ++ X の自由次元 ++ Y の縮約次元（Y の軸の昇順）。
         ;; Y の縮約次元の昇順に並べた X の縮約次元が、結果の末尾の軸に当たる。
         (x-contract-sorted-by-y (mapcar #'cdr (sort (mapcar #'cons y-contract x-contract) #'< :key #'car)))
         (axes-of-result (append x-batch x-kept x-contract-sorted-by-y))
         (perm (loop for i below x-rank collect (position i axes-of-result)))
         (product (%trace-eqn :dot-general (list ct y)
                              :lhs-contracting ans-y :rhs-contracting y-kept
                              :lhs-batch ans-batch :rhs-batch y-batch)))
    (if (equal perm (loop for i below x-rank collect i))
        product
        (%trace-eqn :transpose (list product) :perm perm))))

(def-transpose-rule dot-general (ct invars &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
  ;; 片側が既知で、もう片側が線形。両方が線形なら非線形（接線どうしの積）なのでエラー。
  ;; 右が線形のときは、左右を入れ替えた左の規則（出力の自由次元の並びだけ違う）を使う。
  (destructuring-bind (lhs rhs) invars
    (cond ((and (undefined-primal-p lhs) (undefined-primal-p rhs))
           (error 'autodiff-error
                  :format-control "dot-general の両方の入力が線形な入力のとき、転置できない（接線どうしの積）"))
          ((undefined-primal-p lhs)
           (list (%dot-transpose-lhs ct (undefined-primal-aval lhs) rhs
                                     lhs-contracting rhs-contracting lhs-batch rhs-batch)
                 nil))
          (t
           (list nil
                 (%dot-transpose-lhs ct (undefined-primal-aval rhs) lhs
                                     rhs-contracting lhs-contracting rhs-batch lhs-batch
                                     :swap-ans t))))))

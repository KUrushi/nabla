;;;; ad/rules-shape: 形状演算・縮約・dot-general の jvp ルール（issue #81）。
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

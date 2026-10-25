;;;; ad/rules-gather: gather と scatter-add の jvp / transpose ルール（issue #240）。
;;;;
;;;; どちらも添字を除く入力（gather の operand、scatter-add の updates）について線形で、添字は整数なので
;;;; 接線は常に symbolic zero。jvp は接線に同じ演算を同じ添字でかける。転置は互いに相手になる:
;;;; gather の転置は scatter-add（:size は operand の先頭の軸の長さ）、scatter-add の転置は gather。
;;;; 範囲外の添字の規約（gather は 0、scatter-add は捨てる）は、この転置の組で整合する。

(in-package #:nabla)

(def-jvp-rule gather (primals out tangents)
  (if (symbolic-zero-p (first tangents))
      (make-symbolic-zero (tracer-aval out))
      (%trace-eqn :gather (list (first tangents) (second primals)))))

(def-jvp-rule scatter-add (primals out tangents &key size)
  (if (symbolic-zero-p (second tangents))
      (make-symbolic-zero (tracer-aval out))
      (%trace-eqn :scatter-add (list (first primals) (second tangents)) :size size)))

(defun %gather-check-known-indices (name indices)
  (when (undefined-primal-p indices)
    (error 'autodiff-error
           :format-control "~(~A~) の添字は既知の主値でなければならない（線形な入力にはできない）"
           :format-arguments (list name))))

(def-transpose-rule gather (ct invars)
  (destructuring-bind (operand indices) invars
    (%gather-check-known-indices :gather indices)
    (list (and (undefined-primal-p operand)
               (%trace-eqn :scatter-add (list indices ct)
                           :size (first (aval-shape (undefined-primal-aval operand)))))
          nil)))

(def-transpose-rule scatter-add (ct invars &key size)
  (declare (ignore size))
  (destructuring-bind (indices updates) invars
    (%gather-check-known-indices :scatter-add indices)
    (list nil (and (undefined-primal-p updates) (%trace-eqn :gather (list ct indices))))))

;;;; ad/rules-batch-elementwise: 要素ごとのプリミティブのバッチ化ルール
;;;; （issue #125 は add だけ。残りは #128）。

(in-package #:nabla)

(defun %batch-binary-elementwise (name args batch-dims)
  "形の揃った二項の要素演算 NAME のバッチ化。両方バッチされていれば軸をそろえ、
片方だけなら、バッチされていない側を（形を揃えるために）バッチ軸に沿って複製する。"
  (destructuring-bind (x y) args
    (destructuring-bind (dx dy) batch-dims
      (flet ((size (tracer dim) (nth dim (aval-shape (tracer-aval tracer)))))
        (cond
          ((and dx dy)
           (values (list (%trace-eqn name (list x (%vmap-move-axis y dy dx)))) (list dx)))
          (dx
           (values (list (%trace-eqn name (list x (%vmap-broadcast-batch y dx (size x dx))))) (list dx)))
          (t
           (values (list (%trace-eqn name (list (%vmap-broadcast-batch x dy (size y dy)) y))) (list dy))))))))

(def-batch-rule add (args batch-dims)
  (%batch-binary-elementwise :add args batch-dims))

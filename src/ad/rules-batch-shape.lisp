;;;; ad/rules-batch-shape: 形状プリミティブのバッチ化ルール
;;;; （issue #125 は broadcast-in-dim だけ。残りは #129）。

(in-package #:nabla)

(def-batch-rule broadcast-in-dim (args batch-dims &key shape dims)
  ;; バッチ軸を先頭に動かし、出力でも先頭に置く。
  (let* ((x (%vmap-move-axis (first args) (first batch-dims) 0))
         (size (first (aval-shape (tracer-aval x)))))
    (values (list (%trace-eqn :broadcast-in-dim (list x)
                              :shape (cons size shape)
                              :dims (cons 0 (mapcar #'1+ dims))))
            (list 0))))

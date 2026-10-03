;;;; ad/rules-batch-shape: 形状プリミティブのバッチ化ルール
;;;; （issue #125 が broadcast-in-dim、#129 が reshape・transpose・reduce-sum・reduce-max・dot-general）。

(in-package #:nabla)

(def-batch-rule broadcast-in-dim (args batch-dims &key shape dims)
  ;; バッチ軸を先頭に動かし、出力でも先頭に置く。
  (let* ((x (%vmap-move-axis (first args) (first batch-dims) 0))
         (size (first (aval-shape (tracer-aval x)))))
    (values (list (%trace-eqn :broadcast-in-dim (list x)
                              :shape (cons size shape)
                              :dims (cons 0 (mapcar #'1+ dims))))
            (list 0))))

;;; --- 軸の番号のずらし（バッチ軸 B を挿入した後の番号にする） ---

(defun %shift-axis (axis batch-axis)
  "内側の軸 AXIS が、位置 BATCH-AXIS にバッチ軸を挿入した後で持つ番号。"
  (if (>= axis batch-axis) (1+ axis) axis))

(defun %shift-axes (axes batch-axis)
  (mapcar (lambda (a) (%shift-axis a batch-axis)) axes))

;;; --- reshape / transpose ---

(def-batch-rule reshape (args batch-dims &key shape)
  ;; reshape は row-major で要素を並べ直すので、バッチ軸を先頭に動かしてから
  ;; 形状の先頭にバッチの長さを足す（JAX の _reshape_batch_rule）。
  (let* ((x (%vmap-move-axis (first args) (first batch-dims) 0))
         (size (first (aval-shape (tracer-aval x)))))
    (values (list (%trace-eqn :reshape (list x) :shape (cons size shape)))
            (list 0))))

(def-batch-rule transpose (args batch-dims &key perm)
  ;; バッチ軸を出力でも先頭に置く（JAX の _transpose_batch_rule）。
  (let ((b (first batch-dims)))
    (values (list (%trace-eqn :transpose (list (first args))
                              :perm (cons b (%shift-axes perm b))))
            (list 0))))

;;; --- reduce-sum / reduce-max ---

(defun %batch-reduce (name args batch-dims axes)
  "縮約 NAME のバッチ化。縮約する軸をずらし、出力のバッチ軸は、それより前で
縮約された軸の数だけ前へ詰めた位置になる（JAX の _reducer_batcher）。"
  (let* ((b (first batch-dims))
         (out-dim (- b (count-if (lambda (a) (< a b)) axes))))
    (values (list (%trace-eqn name (list (first args)) :axes (%shift-axes axes b)))
            (list out-dim))))

(def-batch-rule reduce-sum (args batch-dims &key axes)
  (%batch-reduce :reduce-sum args batch-dims axes))

(def-batch-rule reduce-max (args batch-dims &key axes)
  (%batch-reduce :reduce-max args batch-dims axes))

;;; --- dot-general ---

(defun %dot-batch-out-position (batch contracting batch-axis count-before)
  "片側だけがバッチされるときの、出力のバッチ軸の位置。その側の（バッチ軸を挿入した後の）
batch / contracting の番号が BATCH / CONTRACTING、バッチ軸が BATCH-AXIS。
バッチ軸は自由次元になるので、出力では「バッチ次元の個数 + COUNT-BEFORE（出力でこの側の
自由次元より前にある次元の個数）+ この側の自由次元のうち BATCH-AXIS より前の個数」。"
  (+ (length batch) count-before
     (loop for d below batch-axis
           count (not (or (member d batch) (member d contracting))))))

(def-batch-rule dot-general (args batch-dims &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
  (destructuring-bind (lhs rhs) args
    (destructuring-bind (lb rb) batch-dims
      (cond
        ((and lb rb)
         ;; 両側: バッチ軸を先頭に動かし、新しい batch 次元の先頭に加える。
         ;; 出力は「batch 次元 → 自由次元」の先頭が batch 次元なので、出力のバッチ軸は 0。
         (let ((lhs (%vmap-move-axis lhs lb 0))
               (rhs (%vmap-move-axis rhs rb 0)))
           (values (list (%trace-eqn :dot-general (list lhs rhs)
                                     :lhs-contracting (%shift-axes lhs-contracting 0)
                                     :rhs-contracting (%shift-axes rhs-contracting 0)
                                     :lhs-batch (cons 0 (%shift-axes lhs-batch 0))
                                     :rhs-batch (cons 0 (%shift-axes rhs-batch 0))))
                   (list 0))))
        (lb
         ;; lhs だけ: バッチ軸は lhs の自由次元になる。出力の自由次元は lhs 側が先。
         (let* ((lhs-batch* (%shift-axes lhs-batch lb))
                (lhs-contracting* (%shift-axes lhs-contracting lb)))
           (values (list (%trace-eqn :dot-general (list lhs rhs)
                                     :lhs-contracting lhs-contracting*
                                     :rhs-contracting rhs-contracting
                                     :lhs-batch lhs-batch*
                                     :rhs-batch rhs-batch))
                   (list (%dot-batch-out-position lhs-batch* lhs-contracting* lb 0)))))
        (t
         ;; rhs だけ: バッチ軸は rhs の自由次元になる。出力では lhs の自由次元の後ろ。
         (let* ((rhs-batch* (%shift-axes rhs-batch rb))
                (rhs-contracting* (%shift-axes rhs-contracting rb))
                (lhs-free (- (aval-rank (tracer-aval lhs)) (length lhs-batch) (length lhs-contracting))))
           (values (list (%trace-eqn :dot-general (list lhs rhs)
                                     :lhs-contracting lhs-contracting
                                     :rhs-contracting rhs-contracting*
                                     :lhs-batch lhs-batch
                                     :rhs-batch rhs-batch*))
                   (list (%dot-batch-out-position rhs-batch* rhs-contracting* rb lhs-free)))))))))

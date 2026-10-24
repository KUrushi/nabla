;;;; dense-general（issue #208）。Flax の nn.DenseGeneral に合わせる。
;;;;
;;;; - kernel (in... features...)、bias (features...)、:static の contract-axes（縮約する入力の軸の数 n）。
;;;;   入力の最後の n 個の軸を縮約して、特徴の軸に置き換える。multi-head attention の q / k / v は
;;;;   (in) → (heads head-dim)、out は (heads head-dim) → (out)。スロット名は Flax のパラメータ名
;;;;   （kernel / bias）。contract-axes は :static なので葉でなく、チェックポイントのパス名に現れない。
;;;; - 実装は kernel / 入力を平らにして dense と同じ dot にしてから形を戻す（reshape だけで、勾配は
;;;;   reshape と dot の転置から出る）。init は Flax と同じく kernel を平らな形 (prod in, prod features) の
;;;;   lecun_normal で作って reshape する（fan_in は縮約する軸の積）。bias は 0。
;;;; - use_bias（issue #241）: :static の `use-bias`（既定は真）。偽のとき bias は NIL（空の部分木。パス名は
;;;;   kernel のまま）。ほかは layers.lisp の dense と同じ。
;;;; - ロジックは defmodule の外の defun に出す（layers.lisp と同じ理由）。

(in-package #:nabla.nn)

;;; --- dense-general ---

(defun %dense-general-split (name x kernel bias n-in)
  "X・KERNEL・BIAS・N-IN（縮約する軸の数）の形を確かめて (values 先頭の軸 縮約の大きさ 特徴の軸 特徴の大きさ) を返す。"
  (let* ((kernel-shape (%shape kernel))
         (x-shape (%shape x)))
    (unless (and (integerp n-in) (< 0 n-in (length kernel-shape)))
      (error "~A: contract-axes は 1 以上 kernel の rank (~D) 未満の整数でなければならない: ~S"
             name (length kernel-shape) n-in))
    (let ((in-dims (subseq kernel-shape 0 n-in))
          (features (nthcdr n-in kernel-shape)))
      (when (and bias (not (equal features (%shape bias))))
        (error "~A: kernel の形 ~S と bias の形 ~S が合わない（bias は kernel の後ろの ~D 個の軸）"
               name kernel-shape (%shape bias) (length features)))
      (unless (and (>= (length x-shape) n-in) (equal in-dims (last x-shape n-in)))
        (error "~A: 入力の最後の ~D 個の軸が ~S でなければならない（入力の形 ~S）" name n-in in-dims x-shape))
      (values (butlast x-shape n-in) (reduce #'* in-dims) features (reduce #'* features)))))

(defun %dense-general-apply (x kernel bias n-in use-bias)
  (%check-bias-slot "dense-general" use-bias bias)
  (multiple-value-bind (batch in-size features out-size) (%dense-general-split "dense-general" x kernel bias n-in)
    (nabla:reshape
     (%add-bias (funcall (load-time-value
                          (nabla:with-tracing (x kernel x-flat kernel-flat)
                            (nabla:dot (nabla:reshape x x-flat) (nabla:reshape kernel kernel-flat))))
                         x kernel (append batch (list in-size)) (list in-size out-size))
                (and bias (nabla:reshape bias (list out-size))))
     (append batch features))))

(defmodule dense-general (kernel bias (contract-axes :static) (use-bias :static))
  (:init (key in-shape features &key (dtype :f32) (use-bias t))
   (%check-use-bias "dense-general" use-bias)
   (let ((in-shape (if (listp in-shape) in-shape (list in-shape)))
         (features (if (listp features) features (list features))))
     (unless (and in-shape features (every (lambda (n) (and (integerp n) (plusp n))) (append in-shape features)))
       (error "dense-general: in-shape と features は正の整数（か、そのリスト）でなければならない: ~S ~S"
              in-shape features))
     (make-dense-general
      :kernel (nabla:reshape (%lecun-normal key (reduce #'* in-shape) (reduce #'* features) dtype)
                             (append in-shape features))
      :bias (and use-bias (%filled features dtype 0))
      :contract-axes (length in-shape)
      :use-bias use-bias)))
  (:call (self x)
   (%dense-general-apply x (dense-general-kernel self) (dense-general-bias self)
                         (dense-general-contract-axes self) (dense-general-use-bias self))))

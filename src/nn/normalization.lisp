;;;; 正規化のレイヤー: group-norm（issue #260）。
;;;;
;;;; Flax の nn.GroupNorm に合わせる（スロット名 scale / bias はチェックポイントのパス名なので Flax と同じ）。
;;;; 決めたこと:
;;;;
;;;; - 群の決め方: num_groups か group_size のどちらか一方（Flax と同じく num-groups の既定は 32、group-size は
;;;;   NIL。group-size で指定するときは :num-groups nil を明示する。両方・どちらも無しは error）。チャンネル
;;;;   （最後の軸）の数が割り切れなければ error。どちらも :static のスロットにそのまま持ち、群の数は call の
;;;;   ときに入力のチャンネル数から決める（make-group-norm で直接作ったモジュールも同じ検査を通る）。
;;;; - 統計を取る軸: Flax の既定（reduction_axes = None）と同じく、先頭の軸をバッチとみなし、それ以外の軸と
;;;;   群の中のチャンネルで平均と分散を取る（rank 1 の入力はバッチ軸が無く、群の中のチャンネルだけ）。入力を
;;;;   [... 群の数 群の大きさ] に reshape して layer-norm と同じ統計（%layer-norm-statistics。速い分散
;;;;   max(mean(x^2) - mean(x)^2, 0) と rsqrt(var + epsilon)）を取り、形を戻す（Flax の jnp.repeat と同じ）。
;;;;   vmap の中では内側の入力の先頭の軸がバッチになる（Flax と同じ）。Flax の reduction_axes と mask は持たない。
;;;; - y = (x - mean) * (rsqrt(var + epsilon) * scale) + bias（Flax の _normalize と同じ順）。scale / bias の形は
;;;;   (チャンネル数)。use-scale / use-bias は layer-norm と同じ規則（偽なら対応するスロットは NIL）。
;;;; - epsilon の既定は Flax と同じ 1e-6（f64 は 1d-6）。状態は持たない。

(in-package #:nabla.nn)

(defun %check-group-norm-groups (num-groups group-size)
  "NUM-GROUPS と GROUP-SIZE のちょうど一方が正の整数で、もう一方が NIL であることを確かめる。"
  (when (or (and num-groups group-size) (and (null num-groups) (null group-size)))
    (error "group-norm: num-groups と group-size はどちらか一方だけを指定する（group-size で指定するときは :num-groups nil を渡す）: num-groups ~S、group-size ~S"
           num-groups group-size))
  (let ((value (or num-groups group-size)))
    (unless (and (integerp value) (plusp value))
      (error "group-norm: ~A は正の整数でなければならない: ~S" (if num-groups "num-groups" "group-size") value))))

(defun %group-norm-group-count (num-groups group-size channels)
  "チャンネル数 CHANNELS を NUM-GROUPS（か GROUP-SIZE）で分けたときの群の数。割り切れなければ error。"
  (%check-group-norm-groups num-groups group-size)
  (unless (zerop (mod channels (or num-groups group-size)))
    (error "group-norm: チャンネル数 ~D が ~A ~D で割り切れない"
           channels (if num-groups "num-groups" "group-size") (or num-groups group-size)))
  (or num-groups (floor channels group-size)))

(defun %group-norm-reshape (x shape)
  (funcall (load-time-value (nabla:with-tracing (x shape) (nabla:reshape x shape))) x shape))

(defun %group-norm-apply (x scale bias num-groups group-size epsilon use-scale use-bias)
  (%check-optional-slot "group-norm" "use-scale" use-scale "scale" scale)
  (%check-optional-slot "group-norm" "use-bias" use-bias "bias" bias)
  (let* ((shape (%shape x))
         (rank (length shape)))
    (when (zerop rank)
      (error "group-norm: 入力は rank 1 以上（最後の軸がチャンネル）でなければならない: 形 ~S" shape))
    (let* ((channels (car (last shape)))
           (groups (%group-norm-group-count num-groups group-size channels))
           (size (floor channels groups))
           ;; [... 群の数 群の大きさ] の軸で、先頭（バッチ）と群の軸を残し、残りを縮約する（rank 1 は群の軸だけ残す）
           (group-shape (append (butlast shape) (list groups size)))
           (kept-axes (if (= rank 1) '(0) (list 0 (1- rank))))
           (reduce-axes (loop for axis to rank unless (member axis kept-axes) collect axis))
           (count (reduce #'* (mapcar (lambda (axis) (nth axis group-shape)) reduce-axes))))
      (dolist (parameter (list (cons "scale" scale) (cons "bias" bias)))
        (when (and (cdr parameter) (not (equal (list channels) (%shape (cdr parameter)))))
          (error "group-norm: ~A の形は入力のチャンネル数 (~D) でなければならない（~A の形 ~S、入力の形 ~S）"
                 (car parameter) channels (car parameter) (%shape (cdr parameter)) shape)))
      (destructuring-bind (mean inverse-std)
          (%layer-norm-statistics (%group-norm-reshape x group-shape) epsilon reduce-axes kept-axes count)
        (let* ((feature-axes (list (1- rank)))
               (centered (funcall (load-time-value
                                   (nabla:with-tracing (x mean multiplier) (* (- x mean) multiplier)))
                                  x (%group-norm-reshape mean shape)
                                  (%layer-norm-multiplier (%group-norm-reshape inverse-std shape) scale feature-axes))))
          (if bias
              (funcall (load-time-value
                        (nabla:with-tracing (centered bias feature-axes)
                          (+ centered (%broadcast-features bias (%shape centered) feature-axes))))
                       centered bias feature-axes)
              centered))))))

(defun %init-group-norm (features num-groups group-size dtype epsilon use-scale use-bias)
  (unless (and (integerp features) (plusp features))
    (error "group-norm: features（チャンネル数）は正の整数でなければならない: ~S" features))
  (unless (and (realp epsilon) (plusp epsilon))
    (error "group-norm: epsilon は正の実数でなければならない: ~S" epsilon))
  (%check-use-flag "group-norm" "use-scale" use-scale)
  (%check-use-flag "group-norm" "use-bias" use-bias)
  (%group-norm-group-count num-groups group-size features)
  (list :scale (and use-scale (%filled (list features) dtype 1))
        :bias (and use-bias (%filled (list features) dtype 0))
        :num-groups num-groups :group-size group-size :epsilon epsilon :use-scale use-scale :use-bias use-bias))

(defmodule group-norm (scale bias (num-groups :static) (group-size :static) (epsilon :static)
                             (use-scale :static) (use-bias :static))
  (:init (key features &key (num-groups 32) (group-size nil) (dtype :f32) (epsilon (if (eq dtype :f64) 1d-6 1f-6))
                         (use-scale t) (use-bias t))
   (declare (ignore key))
   (apply #'make-group-norm (%init-group-norm features num-groups group-size dtype epsilon use-scale use-bias)))
  (:call (self x)
   (%group-norm-apply x (group-norm-scale self) (group-norm-bias self) (group-norm-num-groups self)
                      (group-norm-group-size self) (group-norm-epsilon self)
                      (group-norm-use-scale self) (group-norm-use-bias self))))

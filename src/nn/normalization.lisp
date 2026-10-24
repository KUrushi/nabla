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
;;;;   [バッチ 空間... 群の数 群の大きさ] に reshape し、[バッチ 群の数 群ごとの要素数] に並べ替えて最後の軸で
;;;;   layer-norm と同じ統計（%layer-norm-statistics。速い分散 max(mean(x^2) - mean(x)^2, 0) と
;;;;   rsqrt(var + epsilon)）を取り、逆の並べ替えで形を戻す（Flax の jnp.repeat と同じ）。最後の軸だけを縮約する
;;;;   のはバックエンドのコンパイラのバグの回避策（%group-norm-statistics）。
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
  "X（配列かトレーサ）を形 SHAPE に reshape する。"
  (funcall (load-time-value (nabla:with-tracing (x shape) (nabla:reshape x shape))) x shape))

(defun %group-norm-transpose (x perm)
  "X を PERM で並べ替える（rank 1・2 の入力では恒等の並べ替えになる）。"
  (funcall (load-time-value (nabla:with-tracing (x perm) (nabla:transpose x perm))) x perm))

(defun %group-norm-layout (shape groups)
  "入力の形 SHAPE（rank 1 以上）を GROUPS 個の群に分けるときの、(values 群に分けた形 並べ替え 統計の形)。
群に分けた形は [バッチ 空間... 群の数 群の大きさ]（rank 1 は [群の数 群の大きさ]）から大きさ 1 の軸を除いたもの、
並べ替えはそれを [バッチ 群の数 空間... 群の大きさ] の順にする perm、統計の形は [バッチ 群の数 群ごとの要素数]
（rank 1 は [群の数 群の大きさ]。群ごとの要素数が 1 なら最後の軸を持たない）。"
  (let* ((rank (length shape))
         (sizes (append (butlast shape) (list groups (floor (car (last shape)) groups))))
         (order (if (= rank 1)
                    '(0 1)
                    (append (list 0 (1- rank)) (loop for axis from 1 below (1- rank) collect axis) (list rank))))
         ;; 大きさ 1 の軸は並べ替えに要らないので除く（並べ替えの eqn を小さくする）
         (kept (or (loop for axis to rank when (> (nth axis sizes) 1) collect axis) (list rank)))
         (perm (loop for axis in order when (member axis kept) collect (position axis kept)))
         (lead (if (= rank 1) (list groups) (list (first shape) groups)))
         (count (floor (reduce #'* shape) (reduce #'* lead))))
    (values (mapcar (lambda (axis) (nth axis sizes)) kept) perm (if (= count 1) lead (append lead (list count))))))

(defun %group-norm-statistics (x groups epsilon)
  "X の群ごとの平均と 1/sqrt(速い分散 + EPSILON) を、X の形に戻した 2 つのリスト。
群の要素を最後の 1 軸に並べ替えてから最後の軸だけを縮約する（layer-norm の既定と同じ形。群の要素が 1 つなら縮約
しない）。先頭のバッチ軸と群の軸の間に空間の軸を挟んだまま縮約すると、jit した grad のコンパイルが
vector.contract の large vector sizes で失敗することがある（CLAUDE.md の group-norm の項、上流のバグ一覧の 9）。"
  (let ((shape (%shape x)))
    (multiple-value-bind (group-shape perm flat-shape) (%group-norm-layout shape groups)
      (let* ((transposed-shape (mapcar (lambda (axis) (nth axis group-shape)) perm))
             (inverse (loop for axis below (length perm) collect (position axis perm)))
             (lead-rank (if (= (length shape) 1) 1 2))
             (reduce-axes (if (> (length flat-shape) lead-rank) (list lead-rank) '())))
        (flet ((back (y)
                 (%group-norm-reshape (%group-norm-transpose (%group-norm-reshape y transposed-shape) inverse) shape)))
          (mapcar #'back
                  (%layer-norm-statistics
                   (%group-norm-reshape (%group-norm-transpose (%group-norm-reshape x group-shape) perm) flat-shape)
                   epsilon reduce-axes (loop for axis below lead-rank collect axis)
                   (if reduce-axes (car (last flat-shape)) 1))))))))

(defun %group-norm-apply (x scale bias num-groups group-size epsilon use-scale use-bias)
  "group-norm の call の本体。X を群ごとに正規化し、SCALE / BIAS（NIL ならかけない・足さない）を適用する。"
  (%check-optional-slot "group-norm" "use-scale" use-scale "scale" scale)
  (%check-optional-slot "group-norm" "use-bias" use-bias "bias" bias)
  ;; make-group-norm で直接作ったモジュールも確かめる（epsilon 0 は分散 0 の群で 0 × inf = NaN になる）
  (%check-group-norm-epsilon epsilon)
  (let* ((shape (%shape x))
         (rank (length shape)))
    (when (zerop rank)
      (error "group-norm: 入力は rank 1 以上（最後の軸がチャンネル）でなければならない: 形 ~S" shape))
    (let* ((channels (car (last shape)))
           (groups (%group-norm-group-count num-groups group-size channels)))
      (dolist (parameter (list (cons "scale" scale) (cons "bias" bias)))
        (when (and (cdr parameter) (not (equal (list channels) (%shape (cdr parameter)))))
          (error "group-norm: ~A の形は入力のチャンネル数 (~D) でなければならない（~A の形 ~S、入力の形 ~S）"
                 (car parameter) channels (car parameter) (%shape (cdr parameter)) shape)))
      (destructuring-bind (mean inverse-std) (%group-norm-statistics x groups epsilon)
        (let* ((feature-axes (list (1- rank)))
               (centered (funcall (load-time-value
                                   (nabla:with-tracing (x mean multiplier) (* (- x mean) multiplier)))
                                  x mean (%layer-norm-multiplier inverse-std scale feature-axes))))
          (if bias
              (funcall (load-time-value
                        (nabla:with-tracing (centered bias feature-axes)
                          (+ centered (%broadcast-features bias (%shape centered) feature-axes))))
                       centered bias feature-axes)
              centered))))))

(defun %check-group-norm-epsilon (epsilon)
  (unless (and (realp epsilon) (plusp epsilon))
    (error "group-norm: epsilon は正の実数でなければならない: ~S" epsilon)))

(defun %init-group-norm (features num-groups group-size dtype epsilon use-scale use-bias)
  "init-group-norm の引数を確かめ、make-group-norm に渡すキーワード引数のリストを返す。"
  (unless (and (integerp features) (plusp features))
    (error "group-norm: features（チャンネル数）は正の整数でなければならない: ~S" features))
  (unless (member dtype '(:f32 :f64))
    (error "group-norm: dtype は :f32 か :f64 でなければならない: ~S" dtype))
  (%check-group-norm-epsilon epsilon)
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

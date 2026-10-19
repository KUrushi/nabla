;;;; conv レイヤー（Flax の nn.Conv。issue #207）。
;;;;
;;;; 初期化と計算は Flax Linen の nn.Conv の既定に合わせる（layers.lisp と同じ方針）。決めたこと:
;;;;
;;;; - 配置は #197 で決めた channels-last: 入力 [N, 空間..., C]（2次元なら NHWC）、kernel
;;;;   [空間..., C/G, O]（HWIO）。スロット名は Flax と同じ kernel / bias（チェックポイントのパス名になる）。
;;;; - kernel の初期値は lecun_normal（layers.lisp の %lecun-normal-of-shape）。fan_in は kernel の面積 ×
;;;;   C/G（Flax の variance_scaling が in 軸の積を fan_in とするのと同じ）。bias は 0。
;;;; - padding は :same / :valid / 明示の ((low high) ...)（空間の次元ごと）。Flax の "SAME" / "VALID"
;;;;   の文字列は受け付けない（キーワードだけ。:static に入る値を EQUAL で比べられるようにし、
;;;;   受け付ける形を増やさないため）。SAME は Flax（JAX の lax.padtype_to_pads）と同じ: 出力の長さ ceil(入力 / stride)、padding の合計
;;;;   max((出力 - 1) stride + 実効 kernel - 入力, 0)、左（low）が合計の半分（切り捨て）、右が残り。
;;;;   実効 kernel = (k - 1) dilation + 1。SAME は入力の形に依存するので呼び出しのたびに計算する。
;;;;   Flax の CIRCULAR / CAUSAL / REFLECT と整数の padding は持たない（必要になったら足す）。
;;;; - stride と kernel-dilation は正の整数か、空間の次元ごとの正の整数のリスト。入力の dilation
;;;;   （転置畳み込み）と use_bias の引数は持たない（dense と同じく、必要になったら足す）。
;;;; - Flax の nn.Conv と同じく空間の次元数は N（kernel-size の長さ。1 次元・2 次元・3 次元…）。call は
;;;;   kernel の rank から N を決める。
;;;; - strides・padding・kernel-dilation は call のたびに正規化する（init と同じ検査）ので、make-conv で
;;;;   直接作ったモジュール（チェックポイントのひな形など）でも `:strides 2` や `:padding :same` が使える。
;;;; - 入力の rank が空間 + 2 ならバッチ付き、空間 + 1 ならバッチ軸の無い入力（Flax と同じ。バッチ1として
;;;;   計算して軸を落とす）。vmap の中では後者の形で書ける。
;;;; - bf16 / f16 の畳み込みの実行系のバグの回避策は convolution の emit 側にある（CLAUDE.md）。
;;;; - ロジックは defmodule の外の defun に出す（layers.lisp と同じ理由）。

(in-package #:nabla.nn)

;;; --- 引数の検査と正規化 ---

(defun %list-length (value)
  "VALUE が（循環していない）真のリストならその長さ、そうでなければ NIL（ドット対も NIL）。"
  (and (listp value) (ignore-errors (list-length value))))

(defun %spatial-list (what value spatial)
  "VALUE（正の整数か、長さ SPATIAL の正の整数のリスト）を長さ SPATIAL のリストにする。"
  (cond ((and (integerp value) (plusp value)) (make-list spatial :initial-element value))
        ((and (eql (%list-length value) spatial) (every (lambda (v) (and (integerp v) (plusp v))) value))
         (copy-list value))
        (t (error "conv: ~A は正の整数か、~D 個の正の整数のリストでなければならない: ~S" what spatial value))))

(defun %normalize-padding (padding spatial)
  "PADDING（:same / :valid / ((low high) ...)）を :same か :valid か、長さ SPATIAL の (low high)
（0 以上の整数）のリストにする。それ以外は error。"
  (cond ((member padding '(:same :valid)) padding)
        ((and (eql (%list-length padding) spatial)
              (every (lambda (pair)
                       (and (eql (%list-length pair) 2) (every (lambda (v) (and (integerp v) (<= 0 v))) pair)))
                     padding))
         (mapcar #'copy-list padding))
        (t (error "conv: padding は :same か :valid か、~D 個の (low high)（0 以上の整数）のリストでなければならない: ~S"
                  spatial padding))))

(defun %same-padding (size kernel stride dilation)
  "SAME の padding (low high)（1つの空間の軸。Flax と同じ。ファイル冒頭を参照）。"
  (let* ((effective (1+ (* (1- kernel) dilation)))
         (output (ceiling size stride))
         (total (max (+ (* (1- output) stride) (- effective size)) 0))
         (low (floor total 2)))
    (list low (- total low))))

(defun %resolve-padding (padding spatial-sizes kernel-sizes strides dilations)
  "正規化した PADDING を、軸ごとの (low high) のリストにする。"
  (case padding
    (:same (mapcar #'%same-padding spatial-sizes kernel-sizes strides dilations))
    (:valid (mapcar (lambda (size) (declare (ignore size)) (list 0 0)) spatial-sizes))
    (t padding)))

;;; --- 計算 ---

(defun %conv-core (x kernel bias strides padding dilations groups)
  "バッチ付きの畳み込み + bias（PADDING は軸ごとの (low high)）。"
  (funcall (load-time-value
            (nabla:with-tracing (x kernel bias strides padding dilations groups)
              (let ((y (nabla:conv-general-dilated x kernel :strides strides :padding padding
                                                            :rhs-dilation dilations
                                                            :feature-group-count groups)))
                (+ y (%broadcast-last bias (%shape y))))))
           x kernel bias strides padding dilations groups))

(defun %add-batch-axis (x)
  (funcall (load-time-value (nabla:with-tracing (x shape) (nabla:reshape x shape))) x (cons 1 (%shape x))))

(defun %drop-batch-axis (y)
  (funcall (load-time-value (nabla:with-tracing (y shape) (nabla:reshape y shape))) y (rest (%shape y))))

(defun %conv-apply (x kernel bias strides padding dilations groups)
  (let* ((kernel-shape (%shape kernel))
         (spatial (- (length kernel-shape) 2))
         (x-shape (%shape x))
         (strides (progn (when (< spatial 1)
                           (error "conv: kernel の rank は 3 以上でなければならない（空間 + 入力 + 出力）: ~S" kernel-shape))
                         (%spatial-list "strides" strides spatial)))
         (dilations (%spatial-list "kernel-dilation" dilations spatial))
         (padding (%normalize-padding padding spatial))
         (unbatched (= (length x-shape) (1+ spatial))))
    (unless (or unbatched (= (length x-shape) (+ spatial 2)))
      (error "conv: 入力の rank は ~D（バッチ付き）か ~D（バッチ軸なし）でなければならない（入力の形 ~S）"
             (+ spatial 2) (1+ spatial) x-shape))
    (unless (= (car (last x-shape)) (* groups (nth spatial kernel-shape)))
      (error "conv: 入力のチャネル数が ~D でなければならない（kernel ~S、グループ数 ~D、入力の形 ~S）"
             (* groups (nth spatial kernel-shape)) kernel-shape groups x-shape))
    (unless (equal (%shape bias) (last kernel-shape))
      (error "conv: bias の形が出力のチャネル数 (~D) と合わない: ~S" (car (last kernel-shape)) (%shape bias)))
    (let* ((batched (if unbatched (%add-batch-axis x) x))
           (sizes (subseq (%shape batched) 1 (1+ spatial)))
           (explicit (%resolve-padding padding sizes (subseq kernel-shape 0 spatial) strides dilations))
           (y (%conv-core batched kernel bias strides explicit dilations groups)))
      (if unbatched (%drop-batch-axis y) y))))

;;; --- モジュール ---

(defmodule conv (kernel bias (strides :static) (padding :static) (kernel-dilation :static)
                          (feature-group-count :static))
  (:init (key in-features out-features kernel-size
          &key (strides 1) (padding :same) (kernel-dilation 1) (feature-group-count 1) (dtype :f32))
   (unless (and (plusp (or (%list-length kernel-size) 0))
                (every (lambda (k) (and (integerp k) (plusp k))) kernel-size))
     (error "conv: kernel-size は1つ以上の正の整数のリストでなければならない: ~S" kernel-size))
   (unless (and (integerp in-features) (plusp in-features) (integerp out-features) (plusp out-features))
     (error "conv: in-features と out-features は正の整数でなければならない: ~S ~S" in-features out-features))
   (unless (and (integerp feature-group-count) (plusp feature-group-count)
                (zerop (mod in-features feature-group-count)) (zerop (mod out-features feature-group-count)))
     (error "conv: feature-group-count は in-features (~D) と out-features (~D) を割り切る正の整数でなければならない: ~S"
            in-features out-features feature-group-count))
   (let ((per-group (floor in-features feature-group-count)))
     (make-conv :kernel (%lecun-normal-of-shape key (append kernel-size (list per-group out-features))
                                                (* (reduce #'* kernel-size) per-group) dtype)
                :bias (%filled (list out-features) dtype 0)
                  :strides (%spatial-list "strides" strides (length kernel-size))
                  :padding (%normalize-padding padding (length kernel-size))
                  :kernel-dilation (%spatial-list "kernel-dilation" kernel-dilation (length kernel-size))
                  :feature-group-count feature-group-count)))
  (:call (self x)
   (%conv-apply x (conv-kernel self) (conv-bias self) (conv-strides self) (conv-padding self)
                  (conv-kernel-dilation self) (conv-feature-group-count self))))

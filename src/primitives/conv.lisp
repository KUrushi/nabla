;;;; conv: convolution プリミティブ（issue #197）。
;;;;
;;;; StableHLO の convolution（JAX の conv_general_dilated）。jvp / transpose /
;;;; バッチ化ルールは src/ad/rules-conv.lisp と src/ad/rules-batch-conv.lisp（issue #203）。
;;;;
;;;; 配置（dimension numbers）は Flax と同じ channels-last に固定する（パラメタには
;;;; しない）。空間次元は 1 以上の任意個（S 個。conv2d なら S=2）:
;;;;   lhs : [N, d_0, ..., d_{S-1}, C]            （NHWC）
;;;;   rhs : [k_0, ..., k_{S-1}, C/G, O]          （HWIO）
;;;;   出力: [N, o_0, ..., o_{S-1}, O]            （NHWC）
;;;; パラメタ:
;;;;   :strides                 長さ S の正の整数のリスト
;;;;   :padding                 長さ S の (low high) のリスト（明示的な値。負も可）
;;;;   :lhs-dilation / :rhs-dilation   長さ S の正の整数のリスト
;;;;   :feature-group-count     G（C と O が G で割り切れる）
;;;;   :batch-group-count       B（既定 1。issue #203。N と O が B で割り切れる。G と同時に 1 より
;;;;                            大きくはできない。lhs のバッチ軸を連続した B 個の塊に分け、g 番目の
;;;;                            出力チャネルの塊（O/B 個）は g 番目のバッチの塊だけを読む。出力のバッチは
;;;;                            N/B。ユーザーが直接使う機能ではなく、rhs の転置（JAX の
;;;;                            _conv_general_dilated_transpose_rhs）が G > 1 のときに現れる）
;;;; v1 で拒否するもの: 整数 dtype、サイズ 0 の次元を持つ
;;;; 入力（N=0 も。JAX は受け付けるが、固定したバックエンドのコンパイラでの空の配列の扱いを確かめていない
;;;; ので v1 では拒否する）、出力の空間サイズが 0 以下になる形。SAME / VALID のような名前つき padding はここでは扱わない
;;;; （呼ぶ側が low / high を計算する）。
;;;;
;;;; 出力の空間サイズは JAX と同じ:
;;;;   floor((dilate(d, lhs) + low + high - dilate(k, rhs)) / stride) + 1
;;;;   ただし dilate(x, r) = (x - 1) * r + 1。
;;;;
;;;; 配置が channels-last 固定なので、転置ルールは明示的な transpose（と rev）を畳み込みの前後に
;;;; 足して JAX の _conv_general_dilated_transpose_lhs / _rhs を写す（src/ad/rules-conv.lisp）。

(in-package #:nabla)

(defun %conv-error (in-avals control &rest args)
  (error 'primitive-error :name :convolution :in-avals in-avals
         :format-control control :format-arguments args))

(defun %conv-check-int-list (in-avals param-name value spatial &key (minimum 1))
  "VALUE が長さ SPATIAL の整数のリストで、各要素が MINIMUM 以上であることを確かめる。"
  (%dot-check-integer-list :convolution in-avals param-name value)
  (unless (= (length value) spatial)
    (%conv-error in-avals "~S の長さは空間次元の数 ~S でなければならない: ~S" param-name spatial value))
  (unless (every (lambda (x) (>= x minimum)) value)
    (%conv-error in-avals "~S の要素は ~S 以上でなければならない: ~S" param-name minimum value)))

(defun %conv-check-padding (in-avals padding spatial)
  (unless (and (sb-int:proper-list-p padding) (= (length padding) spatial)
               (every (lambda (p) (and (sb-int:proper-list-p p) (= (length p) 2) (every #'integerp p)))
                      padding))
    (%conv-error in-avals ":padding は (low high) の組の、長さ ~S のリストでなければならない: ~S"
                 spatial padding)))

(defun %conv-dilate (size dilation)
  "SIZE（1 以上）を DILATION で膨らませた大きさ。"
  (1+ (* (1- size) dilation)))

(defun %conv-out-size (in-size kernel-size stride pad lhs-dilation rhs-dilation)
  "1つの空間次元の出力のサイズ（PAD は (low high)。0 以下になりうる。呼ぶ側が拒否する）。"
  (1+ (floor (- (+ (%conv-dilate in-size lhs-dilation) (first pad) (second pad))
                (%conv-dilate kernel-size rhs-dilation))
             stride)))

(defun %conv-window-text (strides padding lhs-dilation rhs-dilation)
  (flet ((ints (list) (format nil "[~{~D~^, ~}]" list)))
    (format nil "{stride = ~A, pad = [~{[~{~D~^, ~}]~^, ~}], lhs_dilate = ~A, rhs_dilate = ~A}"
            (ints strides) padding (ints lhs-dilation) (ints rhs-dilation))))

(defun %conv-dim-numbers-text (spatial)
  (let ((sp (format nil "~{~D~^, ~}" (loop for i below spatial collect i))))
    (format nil "[b, ~A, f]x[~A, i, o]->[b, ~A, f]" sp sp sp)))

(defun %conv-line (out-name lhs-name rhs-name lhs-type rhs-type out-type
                   strides padding lhs-dilation rhs-dilation groups batch-groups)
  (format nil "~A = stablehlo.convolution(~A, ~A) dim_numbers = ~A, window = ~A {batch_group_count = ~D : i64, feature_group_count = ~D : i64} : (~A, ~A) -> ~A"
          out-name lhs-name rhs-name
          (%conv-dim-numbers-text (length strides))
          (%conv-window-text strides padding lhs-dilation rhs-dilation)
          batch-groups groups lhs-type rhs-type out-type))

(defun %conv-emit-lines (in-names in-avals out-name out-aval strides padding
                         lhs-dilation rhs-dilation groups batch-groups)
  "convolution の :emit 本体。bf16 / f16 は入力を f32 に convert してから畳み込み、
結果を戻す（dot-general の issue #54 と同じく、バックエンドが入力の dtype のまま累積して
eager（single-float 累積）とずれるのを避ける）。"
  (let ((half (member (aval-dtype out-aval) '(:bf16 :f16))))
    (flet ((type-of-aval (aval) (tensor-type-string (if half (make-aval (aval-shape aval) :f32) aval))))
      (let* ((lhs-name (if half (%dot-aux-name "lhs32" out-name) (first in-names)))
             (rhs-name (if half (%dot-aux-name "rhs32" out-name) (second in-names)))
             (acc-name (if half (%dot-aux-name "acc" out-name) out-name))
             (conv (%conv-line acc-name lhs-name rhs-name
                               (type-of-aval (first in-avals)) (type-of-aval (second in-avals))
                               (type-of-aval out-aval)
                               strides padding lhs-dilation rhs-dilation groups batch-groups)))
        (if half
            (format nil "~A~%~A~%~A~%~A"
                    (%dot-convert-line lhs-name (first in-names) (tensor-type-string (first in-avals))
                                       (type-of-aval (first in-avals)))
                    (%dot-convert-line rhs-name (second in-names) (tensor-type-string (second in-avals))
                                       (type-of-aval (second in-avals)))
                    conv
                    (%dot-convert-line out-name acc-name (type-of-aval out-aval)
                                       (tensor-type-string out-aval)))
            conv)))))

(defun %conv-eager (arrays in-avals strides padding lhs-dilation rhs-dilation groups batch-groups)
  "素朴な多重ループによる参照実装。出力の各要素について、カーネルの窓と入力チャネルを
走査して足す。lhs の添字は「膨らませて padding した座標」から逆算し、lhs-dilation の
隙間（割り切れない座標）と範囲外は 0 として読み飛ばす。BATCH-GROUPS が 1 より大きいとき、
出力チャネルの g 番目の塊は lhs のバッチの g 番目の塊（連続した N/B 個）を読む。"
  (let* ((dtype (aval-dtype (first in-avals)))
         (half (member dtype '(:bf16 :f16)))
         (compute-type (if (eq dtype :f64) 'double-float 'single-float))
         (lhs (if half (decode-float16-array (first arrays) dtype) (first arrays)))
         (rhs (if half (decode-float16-array (second arrays) dtype) (second arrays)))
         (lhs-shape (array-dimensions lhs))
         (rhs-shape (array-dimensions rhs))
         (spatial (- (length lhs-shape) 2))
         (in-spatial (subseq lhs-shape 1 (1+ spatial)))
         (kernel-spatial (subseq rhs-shape 0 spatial))
         (batch (floor (first lhs-shape) batch-groups))
         (in-per-group (nth spatial rhs-shape))
         (out-channels (nth (1+ spatial) rhs-shape))
         (out-per-group (floor out-channels groups))
         (out-per-batch-group (floor out-channels batch-groups))
         (out-spatial (loop for d in in-spatial for k in kernel-spatial for s in strides
                            for p in padding for ld in lhs-dilation for rd in rhs-dilation
                            collect (%conv-out-size d k s p ld rd)))
         (result (make-array (append (list batch) out-spatial (list out-channels))
                             :element-type (dtype-element-type dtype)))
         (out-index 0))
    (with-ieee-arithmetic
      (dotimes (n batch)
         (%dot-for-each-index
          out-spatial
          (lambda (out-pos)
            (dotimes (o out-channels)
              (let ((sum (coerce 0 compute-type))
                    (group (floor o out-per-group))
                    (lhs-n (+ (* (floor o out-per-batch-group) batch) n)))
                (%dot-for-each-index
                 kernel-spatial
                 (lambda (k-pos)
                   (let ((in-pos
                           (loop for out in out-pos for k in k-pos for s in strides
                                 for p in padding for ld in lhs-dilation for rd in rhs-dilation
                                 for size in in-spatial
                                 collect (let ((q (+ (- (* out s) (first p)) (* k rd))))
                                           (and (>= q 0) (zerop (mod q ld))
                                                (< (floor q ld) size) (floor q ld))))))
                     (when (every #'identity in-pos)
                       (dotimes (i in-per-group)
                         (incf sum (* (coerce (apply #'aref lhs lhs-n
                                                     (append in-pos (list (+ (* group in-per-group) i))))
                                              compute-type)
                                      (coerce (apply #'aref rhs (append k-pos (list i o)))
                                              compute-type))))))))
                (setf (row-major-aref result out-index)
                      (if half (encode-float16 sum dtype) sum))
                (incf out-index)))))))
    result))

(defprimitive convolution (:strides :padding :lhs-dilation :rhs-dilation :feature-group-count
                           :batch-group-count)
  :abstract-eval
  (lambda (in-avals &key strides padding lhs-dilation rhs-dilation feature-group-count
                      (batch-group-count 1))
    (unless (= (length in-avals) 2)
      (%conv-error in-avals "入力の個数が違う: ~S 個渡されたが 2 個必要" (length in-avals)))
    (let* ((lhs (first in-avals))
           (rhs (second in-avals))
           (lhs-shape (aval-shape lhs))
           (rhs-shape (aval-shape rhs))
           (spatial (- (length lhs-shape) 2)))
      (unless (and (eq (aval-dtype lhs) (aval-dtype rhs)) (%dot-float-dtype-p (aval-dtype lhs)))
        (%conv-error in-avals "convolution は同じ浮動小数点 dtype の2入力が必要: ~S と ~S"
                     (aval-dtype lhs) (aval-dtype rhs)))
      (unless (and (>= spatial 1) (= (length lhs-shape) (length rhs-shape)))
        (%conv-error in-avals "lhs と rhs は同じ rank 3 以上でなければならない: ~S と ~S"
                     lhs-shape rhs-shape))
      (unless (every #'plusp (append lhs-shape rhs-shape))
        (%conv-error in-avals "サイズ 0 の次元を持つ入力は受け付けない: ~S と ~S" lhs-shape rhs-shape))
      (%conv-check-int-list in-avals :strides strides spatial)
      (%conv-check-int-list in-avals :lhs-dilation lhs-dilation spatial)
      (%conv-check-int-list in-avals :rhs-dilation rhs-dilation spatial)
      (%conv-check-padding in-avals padding spatial)
      (unless (typep feature-group-count '(integer 1))
        (%conv-error in-avals ":feature-group-count は正の整数でなければならない: ~S" feature-group-count))
      (unless (typep batch-group-count '(integer 1))
        (%conv-error in-avals ":batch-group-count は正の整数でなければならない: ~S" batch-group-count))
      (when (and (> feature-group-count 1) (> batch-group-count 1))
        (%conv-error in-avals ":feature-group-count ~S と :batch-group-count ~S は同時に 1 より大きくできない"
                     feature-group-count batch-group-count))
      (let ((channels (car (last lhs-shape)))
            (kernel-in (nth spatial rhs-shape))
            (kernel-out (car (last rhs-shape))))
        (unless (and (zerop (mod channels feature-group-count))
                     (= kernel-in (floor channels feature-group-count)))
          (%conv-error in-avals "lhs のチャネル数 ~S は feature-group-count ~S で割り切れ、rhs の入力チャネル数 ~S と C/G が一致しなければならない"
                       channels feature-group-count kernel-in))
        (unless (zerop (mod kernel-out feature-group-count))
          (%conv-error in-avals "rhs の出力チャネル数 ~S は feature-group-count ~S で割り切れなければならない"
                       kernel-out feature-group-count))
        (unless (and (zerop (mod (first lhs-shape) batch-group-count))
                     (zerop (mod kernel-out batch-group-count)))
          (%conv-error in-avals "lhs のバッチ ~S と rhs の出力チャネル数 ~S は batch-group-count ~S で割り切れなければならない"
                       (first lhs-shape) kernel-out batch-group-count))
        (let ((out-spatial (loop for d in (subseq lhs-shape 1 (1+ spatial))
                                 for k in (subseq rhs-shape 0 spatial)
                                 for s in strides for p in padding
                                 for ld in lhs-dilation for rd in rhs-dilation
                                 collect (%conv-out-size d k s p ld rd))))
          (unless (every #'plusp out-spatial)
            (%conv-error in-avals "出力の空間サイズが 0 以下になる: ~S" out-spatial))
          (make-aval (append (list (floor (first lhs-shape) batch-group-count)) out-spatial
                             (list kernel-out))
                     (aval-dtype lhs))))))
  :emit
  (lambda (in-names in-avals out-name out-aval &key strides padding lhs-dilation rhs-dilation
                                                 feature-group-count (batch-group-count 1))
    (%conv-emit-lines in-names in-avals out-name out-aval strides padding
                      lhs-dilation rhs-dilation feature-group-count batch-group-count))
  :eager
  (lambda (arrays in-avals &key strides padding lhs-dilation rhs-dilation feature-group-count
                             (batch-group-count 1))
    (%conv-eager arrays in-avals strides padding lhs-dilation rhs-dilation feature-group-count
                 batch-group-count)))

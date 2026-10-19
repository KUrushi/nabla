;;;; ad/rules-batch-conv: convolution と rev のバッチ化ルール（issue #203）。
;;;;
;;;; JAX の _conv_general_dilated_batch_rule（jax/_src/lax/convolution.py）を、channels-last 固定の
;;;; 配置に写す。バッチ軸を畳み込みの既存の軸に併合して1回の畳み込みにする（vmap のたびに
;;;; 畳み込みを V 回呼ぶのではなく、V 個分を feature group / batch / 出力チャネルに畳む）:
;;;;   - lhs と rhs の両方: lhs のバッチ軸をチャネル軸（batch-group-count > 1 ならバッチ軸）の手前に、
;;;;     rhs のバッチ軸を出力チャネル軸の手前に併合し、グループ数を V 倍にする。出力のチャネル軸 (V, O) を分ける。
;;;;   - lhs だけ: バッチ軸をバッチ軸 N の手前に併合する（batch-group-count > 1 のときはグループの中に
;;;;     V を入れるため、N を (B, N/B) に分けてから (B, V, N/B) の順に併合する）。出力のバッチ軸を分ける。
;;;;   - rhs だけ: バッチ軸を出力チャネル軸の手前に併合する（グループがあるときはグループが外側に
;;;;     来るよう、O を (G, O/G) に分けてから (G, V, O/G) の順に併合する）。出力のチャネル軸を分ける。
;;;; 軸の番号は JAX の lhs_spec / rhs_spec / out_spec に当たるもので、固定の配置では（空間次元の数を
;;;; S として）lhs / 出力のバッチ軸 0、チャネル軸 S+1、rhs の出力チャネル軸 S+1、入力チャネル軸 S。

(in-package #:nabla)

(defun %conv-axis-into (src dst x)
  "JAX の _reshape_axis_into: X の軸 SRC を、SRC を取り除いた後の軸 DST の手前（上位）に併合する
（併合した軸の長さは shape[SRC] * shape[DST]、SRC が外側）。"
  (let* ((shape (aval-shape (tracer-aval x)))
         (rank (length shape))
         (rest (loop for i below rank unless (= i src) collect i))
         (perm (append (subseq rest 0 dst) (list src) (nthcdr dst rest)))
         (new-shape (loop for i in rest for j from 0
                          collect (if (= j dst) (* (nth src shape) (nth i shape)) (nth i shape)))))
    (%conv-trace-reshape (%conv-trace-transpose x perm) new-shape)))

(defun %conv-axis-out-of (src size x)
  "JAX の _reshape_axis_out_of: X の軸 SRC（長さ n）を (SIZE, n / SIZE) の2軸に分ける。"
  (let ((shape (aval-shape (tracer-aval x))))
    (%conv-trace-reshape x (append (subseq shape 0 src)
                                   (list size (floor (nth src shape) size))
                                   (nthcdr (1+ src) shape)))))

(def-batch-rule convolution (args batch-dims &key strides padding lhs-dilation rhs-dilation
                                                  feature-group-count batch-group-count)
  (let* ((spatial (length strides))
         (feature (1+ spatial))            ; lhs / 出力のチャネル軸、rhs の出力チャネル軸
         (lb (first batch-dims))
         (rb (second batch-dims)))
    (flet ((conv (lhs rhs fgc bgc)
             (%conv-trace lhs rhs strides padding lhs-dilation rhs-dilation fgc bgc)))
      (destructuring-bind (lhs rhs) args
        (cond
          ((and lb rb)
           (let* ((size (nth lb (aval-shape (tracer-aval lhs))))
                  (new-rhs (%conv-axis-into rb feature rhs))
                  (out (if (> batch-group-count 1)
                           (conv (%conv-axis-into lb 0 lhs) new-rhs
                                 feature-group-count (* batch-group-count size))
                           (conv (%conv-axis-into lb feature lhs) new-rhs
                                 (* feature-group-count size) batch-group-count))))
             (values (list (%conv-axis-out-of feature size out)) (list feature))))
          (lb
           (let* ((size (nth lb (aval-shape (tracer-aval lhs))))
                  (out (if (= batch-group-count 1)
                           (conv (%conv-axis-into lb 0 lhs) rhs feature-group-count 1)
                           ;; N を (B, N/B) に分けてから V を N/B の手前に、続いて B を V の手前に併合する。
                           (let* ((split (%conv-axis-out-of (if (<= lb 0) 1 0) batch-group-count lhs))
                                  (with-v (%conv-axis-into (+ lb (if (< 0 lb) 1 0)) 1 split)))
                             (conv (%conv-axis-into 0 0 with-v) rhs feature-group-count batch-group-count)))))
             (values (list (%conv-axis-out-of 0 size out)) (list 0))))
          (t
           (let* ((size (nth rb (aval-shape (tracer-aval rhs))))
                  (groups (max feature-group-count batch-group-count)))
             (if (= groups 1)
                 (values (list (%conv-axis-out-of feature size (conv lhs (%conv-axis-into rb feature rhs) 1 1)))
                         (list feature))
                 ;; O を (G, O/G) に分け、V を O/G の手前に、続いて G を V の手前に併合する
                 ;; （グループが外側）。出力のチャネル軸は逆の順に分ける。
                 (let* ((split (%conv-axis-out-of (+ feature (if (<= rb feature) 1 0)) groups rhs))
                        (with-v (%conv-axis-into (+ rb (if (< feature rb) 1 0)) (1+ feature) split))
                        (new-rhs (%conv-axis-into feature feature with-v))
                        (out (conv lhs new-rhs feature-group-count batch-group-count))
                        (out (%conv-axis-out-of feature groups out))
                        (out (%conv-axis-out-of (1+ feature) size out)))
                   (values (list (%conv-axis-into feature (1+ feature) out)) (list feature)))))))))))

(def-batch-rule rev (args batch-dims &key dims)
  ;; バッチ軸の位置はそのままで、反転する軸の番号だけをバッチ軸の挿入後の番号にずらす。
  (values (list (%trace-rev (first args) (%shift-axes dims (first batch-dims))))
          (list (first batch-dims))))

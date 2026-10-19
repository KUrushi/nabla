;;;; ad/rules-conv: convolution と rev の jvp / transpose ルール（issue #203）。
;;;;
;;;; convolution は lhs と rhs の双線形なので、jvp は d(conv(l, r)) = conv(dl, r) + conv(l, dr)
;;;; （ゼロの接線の項は作らない）。transpose は片側が既知、もう片側が線形のときだけ通り、
;;;; JAX の _conv_general_dilated_transpose_lhs / _conv_general_dilated_transpose_rhs
;;;; （jax/_src/lax/convolution.py）を写す。JAX は dimension_numbers で軸の役割を入れ替えて
;;;; 同じ conv_general_dilated を呼ぶが、nabla は配置が channels-last に固定なので、軸の入れ替えを
;;;; 畳み込みの前後の明示的な transpose で行う。
;;;;
;;;; 記号（空間次元の数を S とする）: lhs [N, d_1..d_S, C]、rhs [k_1..k_S, I, O]（I = C/G）、
;;;; 出力 [N, o_1..o_S, O]。
;;;;
;;;; lhs の転置: ct（出力と同じ形）をカーネルを空間反転・入出力入れ替えした rhs と畳み込む。
;;;;   - stride と lhs-dilation を入れ替える（新しい stride = lhs-dilation、新しい lhs-dilation = stride）。
;;;;   - G > 1（feature group）のときは、rhs の出力チャネル O を (G, O/G) に分けて G を入力チャネル軸 I の
;;;;     手前に併合してから I と O を入れ替える（JAX の _reshape_axis_out_of / _reshape_axis_into）。
;;;;     新しい畳み込みの feature-group-count は G。
;;;;   - batch-group-count B > 1 のときも rhs は同じ並べ替えをし、feature-group-count B で畳み込む。
;;;;     出力のチャネル軸 (B, C) をバッチ軸の手前に併合して元の lhs の形 [B*N/B, ..., C] に戻す。
;;;; rhs の転置: lhs の「バッチ軸とチャネル軸を入れ替えたもの」を、ct を「空間 + (N, O)」の
;;;; カーネルとみなして畳み込む。結果は [I, k_1..k_S, O] なので最後に [k.., I, O] へ transpose する。
;;;;   - stride と rhs-dilation を入れ替える（新しい stride = rhs-dilation、新しい rhs-dilation = stride）。
;;;;   - G > 1 のときは batch-group-count = G の畳み込みになり、B > 1 のときは feature-group-count = B になる。

(in-package #:nabla)

;;; --- 軸の並べ替え ---

(defun %conv-trace-transpose (x perm)
  "X を PERM で transpose する（恒等なら eqn を足さない）。"
  (if (equal perm (loop for i below (length perm) collect i))
      x
      (%trace-eqn :transpose (list x) :perm perm)))

(defun %conv-trace-reshape (x shape)
  ;; 呼び出しはすべて軸を分ける・併合するもので rank が変わるので、恒等の reshape にはならない。
  (%trace-eqn :reshape (list x) :shape shape))

(defun %trace-rev (x dims)
  "X の軸 DIMS を反転する。DIMS が空なら何もしない（空の dims の stablehlo.reverse は出さない）。"
  (if dims
      (%trace-eqn :rev (list x) :dims dims)
      x))

(defun %conv-axes (from to)
  (loop for i from from below to collect i))

(defun %conv-rotate-batch-perm (spatial)
  "[N, 空間.., X] → [空間.., N, X] の perm（先頭の軸を空間の後ろへ回す）。
convolution の rhs の転置で ct をカーネルにするときと、結果 [I, k.., O] を [k.., I, O] に
戻すときに使う。"
  (append (%conv-axes 1 (1+ spatial)) (list 0 (1+ spatial))))

(defun %conv-swap-batch-feature-perm (spatial)
  "先頭の軸と最後の軸を入れ替える perm（[N, 空間.., C] → [C, 空間.., N]）。自分自身が逆。"
  (append (list (1+ spatial)) (%conv-axes 1 (1+ spatial)) (list 0)))

(defun %conv-swap-kernel-io-perm (spatial)
  "rhs [k.., I, O] の最後の2軸を入れ替える perm。"
  (append (%conv-axes 0 spatial) (list (1+ spatial) spatial)))

;;; --- padding（JAX の _conv_general_vjp_lhs_padding / _conv_general_vjp_rhs_padding） ---

(defun %conv-dilate-size (size dilation)
  (1+ (* (1- size) dilation)))

(defun %conv-vjp-lhs-padding (in-sizes kernel-sizes strides out-sizes padding lhs-dilation rhs-dilation)
  (loop for in in in-sizes for k in kernel-sizes for s in strides for o in out-sizes
        for p in padding for ld in lhs-dilation for rd in rhs-dilation
        collect (let* ((lhs-dilated (%conv-dilate-size in ld))
                       (rhs-dilated (%conv-dilate-size k rd))
                       (out-dilated (%conv-dilate-size o s))
                       (before (- rhs-dilated (first p) 1))
                       (after (- (+ lhs-dilated rhs-dilated) 1 out-dilated before)))
                  (list before after))))

(defun %conv-vjp-rhs-padding (in-sizes kernel-sizes strides out-sizes padding lhs-dilation rhs-dilation)
  (loop for in in in-sizes for k in kernel-sizes for s in strides for o in out-sizes
        for p in padding for ld in lhs-dilation for rd in rhs-dilation
        collect (let* ((lhs-dilated (%conv-dilate-size in ld))
                       (rhs-dilated (%conv-dilate-size k rd))
                       (out-dilated (%conv-dilate-size o s)))
                  (list (first p)
                        (+ (- out-dilated lhs-dilated) (- rhs-dilated (first p) 1))))))

(defun %conv-trace (lhs rhs strides padding lhs-dilation rhs-dilation feature-group-count batch-group-count)
  (%trace-eqn :convolution (list lhs rhs)
              :strides strides :padding padding :lhs-dilation lhs-dilation :rhs-dilation rhs-dilation
              :feature-group-count feature-group-count :batch-group-count batch-group-count))

;;; --- jvp ---

(def-jvp-rule convolution (primals out tangents &key strides padding lhs-dilation rhs-dilation
                                                     feature-group-count batch-group-count)
  ;; 積の微分: d(conv(l, r)) = conv(dl, r) + conv(l, dr)。ゼロの接線の項は作らない。
  (flet ((conv (lhs rhs)
           (%conv-trace lhs rhs strides padding lhs-dilation rhs-dilation
                        feature-group-count batch-group-count)))
    (destructuring-bind (lhs rhs) primals
      (destructuring-bind (lhs-tangent rhs-tangent) tangents
        (add-tangents
         (if (symbolic-zero-p lhs-tangent)
             (make-symbolic-zero (tracer-aval out))
             (conv lhs-tangent rhs))
         (if (symbolic-zero-p rhs-tangent)
             (make-symbolic-zero (tracer-aval out))
             (conv lhs rhs-tangent)))))))

(def-jvp-rule rev (primals out tangents &key dims)
  (declare (ignore primals out))
  ;; 軸の反転は線形なので、接線にそのままかける。
  (%trace-rev (first tangents) dims))

;;; --- transpose ---

(defun %conv-transpose-lhs (ct rhs lhs-aval strides padding lhs-dilation rhs-dilation
                            feature-group-count batch-group-count)
  "lhs が線形のときの lhs の余接線（JAX の _conv_general_dilated_transpose_lhs）。"
  (let* ((spatial (length strides))
         (rhs-shape (aval-shape (tracer-aval rhs)))
         (kernel-sizes (subseq rhs-shape 0 spatial))
         (in-sizes (subseq (aval-shape lhs-aval) 1 (1+ spatial)))
         (out-sizes (subseq (aval-shape (tracer-aval ct)) 1 (1+ spatial)))
         (groups (max feature-group-count batch-group-count))
         (kernel-in (nth spatial rhs-shape))
         (kernel-out (nth (1+ spatial) rhs-shape))
         (reversed (%trace-rev rhs (%conv-axes 0 spatial)))
         ;; 出力チャネル O を (G, O/G) に分け、G を入力チャネル I の手前に併合し、I と O を入れ替える:
         ;; [k.., I, O] → [k.., I, G, O/G] → [k.., O/G, G, I] → [k.., O/G, G*I]。
         (kernel (if (= groups 1)
                     (%conv-trace-transpose reversed (%conv-swap-kernel-io-perm spatial))
                     (let* ((split (%conv-trace-reshape
                                    reversed (append (subseq rhs-shape 0 spatial)
                                                     (list kernel-in groups (floor kernel-out groups)))))
                            (moved (%conv-trace-transpose
                                    split (append (%conv-axes 0 spatial)
                                                  (list (+ spatial 2) (+ spatial 1) spatial)))))
                       (%conv-trace-reshape
                        moved (append (subseq rhs-shape 0 spatial)
                                      (list (floor kernel-out groups) (* groups kernel-in)))))))
         (conv (%conv-trace ct kernel
                            lhs-dilation
                            (%conv-vjp-lhs-padding in-sizes kernel-sizes strides out-sizes
                                                   padding lhs-dilation rhs-dilation)
                            strides rhs-dilation groups 1)))
    (if (= batch-group-count 1)
        conv
        ;; 出力のチャネル軸 (B, C) をバッチ軸の手前へ併合する: [N', 空間.., B*C] → [B*N', 空間.., C]。
        (let* ((conv-shape (aval-shape (tracer-aval conv)))
               (split (%conv-trace-reshape
                       conv (append (list (first conv-shape)) (subseq conv-shape 1 (1+ spatial))
                                    (list batch-group-count (floor (car (last conv-shape)) batch-group-count)))))
               (moved (%conv-trace-transpose
                       split (append (list (1+ spatial)) (%conv-axes 0 (1+ spatial)) (list (+ spatial 2))))))
          (%conv-trace-reshape moved (aval-shape lhs-aval))))))

(defun %conv-transpose-rhs (ct lhs rhs-aval strides padding lhs-dilation rhs-dilation
                            feature-group-count batch-group-count)
  "rhs が線形のときの rhs の余接線（JAX の _conv_general_dilated_transpose_rhs）。"
  (let* ((spatial (length strides))
         (in-sizes (subseq (aval-shape (tracer-aval lhs)) 1 (1+ spatial)))
         (kernel-sizes (subseq (aval-shape rhs-aval) 0 spatial))
         (out-sizes (subseq (aval-shape (tracer-aval ct)) 1 (1+ spatial)))
         ;; lhs はバッチ軸とチャネル軸を入れ替えて（チャネルがバッチ、バッチがチャネルになる）、
         ;; ct は [N, 空間.., O] を [空間.., N, O] のカーネルにする。
         (lhs* (%conv-trace-transpose lhs (%conv-swap-batch-feature-perm spatial)))
         (kernel (%conv-trace-transpose ct (%conv-rotate-batch-perm spatial)))
         (conv (%conv-trace lhs* kernel
                            rhs-dilation
                            (%conv-vjp-rhs-padding in-sizes kernel-sizes strides out-sizes
                                                   padding lhs-dilation rhs-dilation)
                            lhs-dilation strides
                            ;; G > 1 は batch-group-count = G に、B > 1 は feature-group-count = B になる。
                            batch-group-count feature-group-count)))
    ;; 結果は [I, k.., O]。rhs の並び [k.., I, O] に戻す。
    (%conv-trace-transpose conv (%conv-rotate-batch-perm spatial))))

(def-transpose-rule convolution (ct invars &key strides padding lhs-dilation rhs-dilation
                                                feature-group-count batch-group-count)
  (destructuring-bind (lhs rhs) invars
    (cond ((and (undefined-primal-p lhs) (undefined-primal-p rhs))
           (error 'autodiff-error
                  :format-control "convolution の両方の入力が線形な入力のとき、転置できない（接線どうしの積）"))
          ((undefined-primal-p lhs)
           (list (%conv-transpose-lhs ct rhs (undefined-primal-aval lhs) strides padding
                                      lhs-dilation rhs-dilation feature-group-count batch-group-count)
                 nil))
          ((undefined-primal-p rhs)
           (list nil
                 (%conv-transpose-rhs ct lhs (undefined-primal-aval rhs) strides padding
                                      lhs-dilation rhs-dilation feature-group-count batch-group-count)))
          (t
           (error 'autodiff-error
                  :format-control "convolution の転置: 線形な入力が無い（両方とも既知の値）")))))

(def-transpose-rule rev (ct invars &key dims)
  (declare (ignore invars))
  ;; 軸の反転は自分自身が逆（置換行列は対称）。
  (list (%trace-rev ct dims)))

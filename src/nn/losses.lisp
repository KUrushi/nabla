;;;; 損失関数: 教師のロジットを取る交差エントロピー（Softmax Cancellation の修正）。
;;;;
;;;; 蒸留や KL 正則化の交差エントロピー -sum(p * log-softmax(z_s))（p = softmax(z_t)）の勾配は
;;;; q - p（q = softmax(z_s)）。素朴に自動微分すると、ほぼ等しい確率 q と p を別々に丸めてから
;;;; 引くので桁を失い、さらに順伝播の softmax と逆伝播の log-softmax の式の違いで、z_s = z_t でも
;;;; 勾配が 0 にならない（Barron ら「A Fix for Softmax and Cross-Entropy」の Softmax Cancellation。
;;;; docs/numerics.md）。Adam はこの丸め誤差を学習率の大きさの更新に増幅する（生徒が教師から離れる）。
;;;; ここでは論文の 3 節と同じく、ロジットの差から q - p を対数空間で直接計算し、stop-gradient で
;;;; 勾配を差し替える。

(in-package #:nabla.nn)

(defun %soft-target-cross-entropy (z-s z-t axis)
  "AXIS（0 以上）に沿った soft-target 交差エントロピー（AXIS を潰した形）。論文の式 (9):
d = delta - sum(p delta)、s = log(q/p) = d - log1p(sum(p expm1(d)))、u = q - p = p expm1(s)
（|s| >= 1 と p = 0 では q - p にする。p が 0 に丸まって expm1 が溢れると 0 * inf になるため）、
勾配 g = u - q sum(u)（和が 0）。教師の確率が 0 の位置（-inf のロジット）があると sum(p exp(delta))
から生徒のその位置の確率の和 Q_u が抜けるので、s に log1p(-Q_u) を足す（論文の式に無い補正）。

値は -sum(p log q)。勾配は stop-gradient で g に差し替える。2階微分（ヘッセ行列 diag(q) - q q^T）も
正しくなるよう、値が 0 で1階微分も 0 の項 (q - sg(q))(z_s - sg(z_s)) / 2 を足す（この項の2階微分は
(dq_j/dz_i + dq_i/dz_j) / 2 = H_ij。3 階以上は正しくない）。教師 Z-T には勾配を流さない。"
  (funcall (load-time-value
            (nabla:with-tracing (z-s z-t axis)
              (let* ((shape (%shape z-s))
                     (axes (list axis))
                     (z-t (nabla:stop-gradient z-t))
                     (log-q (%log-softmax z-s axis))
                     (q (exp log-q))
                     (p (%softmax z-t axis))
                     (taught (> p 0))
                     ;; 教師の確率が 0 の位置（-inf のロジット）は差を 0 にする（-inf - -inf = NaN を避ける）
                     (delta (nabla:where taught (- z-s z-t) 0))
                     (d (- delta (%expand (nabla:reduce-sum (* p delta) :axes axes) shape axis)))
                     ;; 教師の確率が 0 の位置にある生徒の確率 Q_u は sum(p exp(delta)) に入らないので、
                     ;; log(q/p) に log1p(-Q_u) を足して補う（論文の式は p > 0 の行だけを考えている）
                     (untaught (nabla:reduce-sum (nabla:where taught 0 q) :axes axes))
                     (s (- d (%expand (- (nabla:log1p (nabla:reduce-sum (* p (nabla:expm1 d)) :axes axes))
                                         (nabla:log1p (- untaught)))
                                      shape axis)))
                     (u (nabla:where (< (nabla:where taught (max s (- s)) 2) 1) (* p (nabla:expm1 s)) (- q p)))
                     (g (nabla:stop-gradient (- u (* q (%expand (nabla:reduce-sum u :axes axes) shape axis)))))
                     ;; 値が 0 で勾配が 1 の項。±inf のロジットでは inf - inf = NaN になるので 0 にする（勾配も 0）
                     (perturbation (nabla:where (= (- z-s z-s) 0) (- z-s (nabla:stop-gradient z-s)) 0))
                     (loss (- (nabla:reduce-sum (nabla:where taught (* p (nabla:stop-gradient log-q)) 0) :axes axes))))
                (+ loss (nabla:reduce-sum (* (+ g (* 0.5d0 (- q (nabla:stop-gradient q)))) perturbation) :axes axes)))))
           z-s z-t axis))

(defun soft-target-cross-entropy (logits teacher-logits &key (axis -1))
  "教師のロジット TEACHER-LOGITS の softmax を目標にした、LOGITS の交差エントロピー
-sum(softmax(TEACHER-LOGITS) * log-softmax(LOGITS))（AXIS を潰した形。既定は最後の軸）。蒸留と
KL 正則化に使う。LOGITS についての勾配 softmax(LOGITS) - softmax(TEACHER-LOGITS) を、2つの
確率を引かずにロジットの差から計算するので、LOGITS が TEACHER-LOGITS に近くても桁を失わず、等しい
ときはちょうど 0（Softmax Cancellation を起こさない。docs/numerics.md）。

TEACHER-LOGITS は定数として扱い、勾配を流さない。LOGITS についての2階微分（ヘッセ行列）までは正しく、
3 階以上は正しくない。TEACHER-LOGITS が -inf の位置は目標の確率 0 として扱い、LOGITS も -inf ならその位置の損失と
勾配は 0（NaN を出さない）。
LOGITS と TEACHER-LOGITS は同じ形・dtype の配列かトレーサ。"
  (unless (equal (%shape logits) (%shape teacher-logits))
    (error "soft-target-cross-entropy: LOGITS と TEACHER-LOGITS の形が違う: ~S と ~S"
           (%shape logits) (%shape teacher-logits)))
  (%soft-target-cross-entropy logits teacher-logits (%resolve-axis axis logits)))

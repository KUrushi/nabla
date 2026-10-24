;;;; 提案: nabla.nn に sigmoid と、その「安定な」版を足す。

(defun sigmoid (x)
  "1 / (1 + exp(-x))（要素ごと）。勾配は自動微分で sigmoid(x) (1 - sigmoid(x)) になる。"
  (funcall (load-time-value (nabla:with-tracing (x) (/ 1 (+ 1 (exp (- x)))))) x))

(defun stable-sigmoid (x)
  "exp が溢れないよう |x| で分ける版。|x| は max(x, -x)。jit しても eager と同じ値になる。"
  (funcall (load-time-value
            (nabla:with-tracing (x)
              (let* ((e (exp (- (max x (- x)))))
                     (r (/ 1 (+ 1 e))))
                (nabla:where (>= x 0) r (* e r)))))
           x))

;;;; テスト（PBT、small）
(def-layer-property layers/sigmoid-matches-reference
  "sigmoid と stable-sigmoid の値と勾配は double の 1/(1+exp(-x)) と一致する。"
  (seed) (uniform-integer :lo 0 :hi 100000)
  (let ((x (%rand '(4 5) seed)))   ; 要素は [-1, 1)、f32
    (and (allclose (stable-sigmoid x) (sigmoid x) :dtype :f32 :rtol 1d-5 :atol 1d-6)
         (allclose (funcall (nabla:grad (nabla:with-tracing (x) (nabla:reduce-sum (stable-sigmoid x)))) x)
                   (funcall (nabla:grad (nabla:with-tracing (x) (nabla:reduce-sum (sigmoid x)))) x)
                   :dtype :f32 :rtol 1d-5 :atol 1d-6))))

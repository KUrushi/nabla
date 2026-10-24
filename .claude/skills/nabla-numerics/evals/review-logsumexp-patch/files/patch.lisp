;;;; 提案: nabla.nn に logsumexp と、それを使う softmax 交差エントロピーを足す。

(defun logsumexp (x &key (axis -1))
  "AXIS に沿った log(sum(exp(X)))。最大値を引いてから exp するので溢れない。
最大値は stop-gradient で定数扱い（引いても微分は変わらない）。"
  (let ((axis (%resolve-axis axis x)))
    (funcall (load-time-value
              (nabla:with-tracing (x axis)
                (let* ((m (nabla:stop-gradient (nabla:reduce-max x :axes (list axis))))
                       (shifted (- x (%expand m (%shape x) axis))))
                  (+ m (log (nabla:reduce-sum (exp shifted) :axes (list axis)))))))
             x axis)))

(defun softmax-cross-entropy (logits labels)
  "-sum(LABELS * (LOGITS - logsumexp(LOGITS)))（最後の軸）。LABELS は確率。"
  (funcall (load-time-value
            (nabla:with-tracing (logits labels)
              (- (nabla:reduce-sum (* labels (- logits (%expand (logsumexp logits) (%shape logits) (1- (length (%shape logits))))))
                                   :axes (list (1- (length (%shape logits))))))))
           logits labels))

;;;; テスト（PBT、small）
(def-layer-property losses/logsumexp-matches-reference
  "logsumexp は log(sum(exp(x))) を double で計算したものと一致し、softmax-cross-entropy の勾配は
softmax(x) - labels と一致する。"
  (seed) (uniform-integer :lo 0 :hi 100000)
  (let* ((x (%rand '(4 5) seed))          ; 要素は [-1, 1)
         (labels (softmax (%rand '(4 5) (1+ seed)))))
    (and (allclose (logsumexp x) (naive-double-logsumexp x) :dtype :f32 :rtol 1d-5 :atol 1d-6)
         (allclose (funcall (nabla:grad (nabla:with-tracing (x) (nabla:reduce-sum (softmax-cross-entropy x labels)))) x)
                   (naive-double-softmax-minus labels x)
                   :dtype :f32 :rtol 1d-5 :atol 1d-6))))

;;;; multi-head attention を IREE の local で jit する（issue #208、medium）。
;;;;
;;;; 守らせる性質:
;;;; - jit した順伝播と jit した grad が、Flax のフィクスチャ（attention-test.lisp と同じ全ケース）と一致する。
;;;; - dropout は jit しても eager と同じ出力（同じキー → 同じマスク）。deterministic は静的引数。
;;;; - init で作ったモデルを因果マスクつきで jit して eager と一致する（use-bias が偽のモデルも、issue #241）。
;;;; 入力は 1 回の jit で 162 個まで（IREE local-task の制限、CLAUDE.md）。ここのモデルは葉が 8 個。

(in-package #:nabla.nn.tests)

(in-suite :nabla.medium)

(test attention/jit-forward-and-grads-match-flax-on-iree-local
  "attention-test.lisp のフィクスチャの全ケースで、IREE の local で jit した順伝播と jit した grad が
Flax の出力・勾配と一致する（許容誤差は eager と同じ）。"
  (when (%layers-require-iree-or-skip)
    (let ((backend (nabla:find-backend :iree))
          (nabla:*compile-cache-directory* nil))
      (dolist (name (append *attention-case-names* *attention-no-bias-case-names*))
        (let ((case (%att-case name)))
          (multiple-value-bind (loss forward argnums) (%att-loss-and-forward case)
            (let* ((model (%att-model case))
                   (inputs (%att-inputs case))
                   (ct (%fx-array case "ct"))
                   (jf (nabla:jit forward :backend backend))
                   (jg (nabla:jit (nabla:grad loss :argnums argnums) :backend backend)))
              (unwind-protect
                   (let* ((grads (apply jg model (append inputs (list ct))))
                          (expected (%att-expected-grads case))
                          (actual (append (nabla:tree-leaves (first grads)) (rest grads))))
                     (is (%close-f32 (apply jf model inputs) (%fx-array case "out"))
                         "~A: jit した順伝播が Flax と一致しない" name)
                     (is (= (length expected) (length actual)) "~A: 勾配の個数" name)
                     (loop for a in actual for e in expected for leaf in (%att-grad-names case)
                           do (%att-check-leaf name leaf a e)))
                (nabla::%jit-cache-forget forward)
                (nabla::%jit-cache-forget loss)))))))))

(test attention/dropout-jit-matches-eager-and-deterministic-is-static
  "attention の dropout を jit した出力は eager と同じ（同じキーなら同じマスク）。deterministic を静的引数にすると、
真のとき dropout なしの出力、偽のとき dropout がかかる。"
  (when (%layers-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (base (init-multi-head-attention (nabla:prng-key 3) 2 8))
           (model (make-multi-head-attention :query (multi-head-attention-query base)
                                             :key (multi-head-attention-key base)
                                             :value (multi-head-attention-value base)
                                             :out (multi-head-attention-out base) :dropout-rate 0.3f0))
           (f (nabla:with-tracing (m x mask key deterministic) (call m x x mask key deterministic)))
           (jf (nabla:jit f :backend backend :static-args '(4)))
           (x (%rand '(2 6 8) 5))
           (mask (make-causal-mask 6))
           (key (nabla:prng-key 77)))
      (unwind-protect
           (let ((plain (call model x x mask nil t)))
             (is (%close-f32 (funcall jf model x mask key nil) (call model x x mask key nil)))
             (is (not (equalp (funcall jf model x mask key nil) plain)))
             (is (%close-f32 (funcall jf model x mask key t) plain)))
        (nabla::%jit-cache-forget f)))))

(test attention/jit-of-an-initialized-causal-model-matches-eager
  "init-multi-head-attention で作ったモデル（4 ヘッド、幅 16）を因果マスクつきで jit した出力と勾配
\(モデルと同じ構造）が eager と一致する。"
  (when (%layers-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (model (init-multi-head-attention (nabla:prng-key 9) 4 16))
           (x (%rand '(3 5 16) 8))
           (mask (make-causal-mask 5))
           (f (nabla:with-tracing (m x mask) (call m x x mask nil t)))
           (loss (nabla:with-tracing (m x mask) (nabla:reduce-sum (call m x x mask nil t))))
           (g (nabla:grad loss))
           (jf (nabla:jit f :backend backend))
           (jg (nabla:jit g :backend backend)))
      (unwind-protect
           (let ((jit-grad (funcall jg model x mask)) (eager-grad (funcall g model x mask)))
             (is (%close-f32 (funcall jf model x mask) (funcall f model x mask)))
             (is (typep jit-grad 'multi-head-attention))
             (is (every #'%close-f32 (nabla:tree-leaves jit-grad) (nabla:tree-leaves eager-grad))))
        (nabla::%jit-cache-forget f)
        (nabla::%jit-cache-forget g)))))

(test attention/half-precision-with-a-causal-mask-works-under-jit-and-matches-f32
  "bf16 と f16 のモデルと入力で、因果マスクつきの attention を IREE の local で jit できる（マスクで隠す値は
その dtype の最小の有限値: bf16 は -3.39e38、f16 は -65504。Flax の finfo(dtype).min）。出力は有限で、同じ
パラメータの f32 モデルの eager の出力と一致する（仮数が 8 ビット（bf16）・11 ビット（f16）で、射影・softmax・
内積を重ねるので rtol / atol は 5e-2。f32 同士の 1e-5 ではない）。マスクを 0 にしたり落としたりすると、隠した
位置が混ざって出力が大きくずれる。"
  (when (%layers-require-iree-or-skip)
    (dolist (dtype '(:bf16 :f16))
      (let* ((backend (nabla:find-backend :iree))
             (nabla:*compile-cache-directory* nil)
             (model (init-multi-head-attention (nabla:prng-key 21) 2 8))
             (x (%rand '(2 5 8) 22))
             (mask (make-causal-mask 5))
             (to-half (lambda (array)
                        (nabla:to-device (nabla::encode-float16-array array dtype) backend :dtype dtype)))
             (half-model (nabla:tree-map to-half model))
             (half-x (funcall to-half x))
             (f (nabla:with-tracing (m x mask) (call m x x mask nil t)))
             (jf (nabla:jit f :backend backend)))
        (unwind-protect
             (let ((out (nabla::decode-float16-array (funcall jf half-model half-x mask) dtype))
                   (expected (call model x x mask nil t)))
               (is (every (lambda (v) (= v v)) (%elements out)) "~A: NaN が無い" dtype)
               (is (allclose out expected :dtype :f32 :rtol 5d-2 :atol 5d-2) "~A: f32 と一致する" dtype))
          (nabla::%jit-cache-forget f))))))

(test attention/jit-of-a-model-without-bias-matches-eager
  "init-multi-head-attention :use-bias nil（bias の無い 4 つの射影）のモデルを因果マスクつきで jit した出力と勾配
\(葉は 4 つの kernel だけ）が eager と一致する。"
  (when (%layers-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (model (init-multi-head-attention (nabla:prng-key 9) 4 16 :use-bias nil))
           (x (%rand '(3 5 16) 8))
           (mask (make-causal-mask 5))
           (f (nabla:with-tracing (m x mask) (call m x x mask nil t)))
           (loss (nabla:with-tracing (m x mask) (nabla:reduce-sum (call m x x mask nil t))))
           (g (nabla:grad loss))
           (jf (nabla:jit f :backend backend))
           (jg (nabla:jit g :backend backend)))
      (unwind-protect
           (let ((jit-grad (funcall jg model x mask)) (eager-grad (funcall g model x mask)))
             (is (%close-f32 (funcall jf model x mask) (funcall f model x mask)))
             (is (= 4 (length (nabla:tree-leaves jit-grad))))
             (is (every #'%close-f32 (nabla:tree-leaves jit-grad) (nabla:tree-leaves eager-grad))))
        (nabla::%jit-cache-forget f)
        (nabla::%jit-cache-forget g)))))

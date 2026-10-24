;;;; Softmax Collapse / Softmax Cancellation の修正が IREE の local で jit しても成り立つ（medium）。
;;;;
;;;; softmax-numerics-test.lisp と losses-test.lisp の性質のうち、バックエンドの数学関数（exp・
;;;; log_plus_one・exponential_minus_one の近似）と融合に左右されうるものを jit で確かめる:
;;;; - 余裕（最大値と他の差）が大きい行の交差エントロピーの勾配が 0 に丸まらず、参照と一致する
;;;;   （f32 と bf16）。
;;;; - soft-target-cross-entropy の勾配は z_s = z_t でちょうど 0、論文の例で厳密な値と一致する。

(in-package #:nabla.nn.tests)

(in-suite :nabla.medium)

(test softmax-numerics/jit-on-iree-keeps-the-target-gradient
  "IREE の local で jit した交差エントロピーの勾配は、余裕 9〜80（f32）・6〜30（bf16）の行で
double の参照と一致する（f32 は相対 1e-4、bf16 は bf16 に丸めた入力の参照と相対 2e-2）。"
  (when (%layers-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (ce (nabla:with-tracing (x onehot) (- (nabla:reduce-sum (* (log-softmax x) onehot) :axes '(0)))))
           (jg (nabla:jit (nabla:grad ce) :backend backend)))
      (unwind-protect
           (progn
             (dolist (margin '(9 20 50 80))
               (let* ((x (%f32-vector (list margin 0 -1 0.5)))
                      (g (funcall jg x (%one-hot 0 4))))
                 (is (%close-relative-p g (%reference-log-softmax-vjp x #(-1 0 0 0)) 1d-4)
                     "f32 余裕 ~D: ~S" margin g)))
             (dolist (margin '(6 10 20 30))
               (let* ((bits (nabla:convert (%f32-vector (list margin 0 -1 0.5)) :bf16))
                      (g (nabla::decode-float16-array
                          (funcall jg (nabla:to-device bits backend :dtype :bf16)
                                   (nabla:to-device (nabla:convert (%one-hot 0 4) :bf16) backend :dtype :bf16))
                          :bf16)))
                 (is (%close-relative-p g (%reference-log-softmax-vjp (nabla::decode-float16-array bits :bf16)
                                                                      #(-1 0 0 0))
                                        2d-2)
                     "bf16 余裕 ~D: ~S" margin g))))
        (nabla::%jit-cache-forget ce)))))

(test losses/jit-on-iree-gives-exact-zero-at-equality
  "IREE の local で jit した soft-target-cross-entropy の勾配は、z_s = z_t でちょうど 0、論文の例
（z_t = [2, 1, 0]、z_s = z_t + [-1e-7, 1e-7, 0]）で厳密な値と相対誤差 1e-3 で一致する。"
  (when (%layers-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (soft (%soft-ce))
           (jg (nabla:jit (nabla:grad soft) :backend backend)))
      (unwind-protect
           (let* ((zt (%f32-vector '(2 1 0)))
                  (zs (let ((v (copy-seq zt))) (decf (aref v 0) 1e-7) (incf (aref v 1) 1e-7) v))
                  (z (%dominated-row 11 6 2 25)))
             (is (every #'zerop (funcall jg z z)))
             (is (%close-relative-p (funcall jg zs zt) #(-4.596d-8 4.144d-8 4.513d-9) 1d-3)))
        (nabla::%jit-cache-forget soft)))))

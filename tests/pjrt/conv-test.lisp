;;;; convolution プリミティブ（issue #197）を PJRT (CPU) で jit した結果が、JAX の
;;;; conv_general_dilated のフィクスチャと一致することを確かめる。

(in-package #:nabla.pjrt.tests)

(define-pjrt-test conv/jit-pjrt-matches-jax-fixture
  "JAX の conv_general_dilated のフィクスチャ（1〜3 次元、stride・padding（負も）・膨張・グループ）を
conv-general-dilated で書いて PJRT (CPU) で jit した結果は、JAX の出力と f32 の許容誤差で一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (nabla:find-backend :pjrt))
        (nb:*compile-cache-directory* nil))
    (dolist (case (read-conv-fixture))
      (destructuring-bind (&key name strides padding lhs-dilation rhs-dilation groups lhs rhs out) case
        (let* ((f (nb:with-tracing (x w)
                    (nb:conv-general-dilated x w :strides strides :padding padding
                                                 :lhs-dilation lhs-dilation :rhs-dilation rhs-dilation
                                                 :feature-group-count groups)))
               (jf (nb:jit f :backend backend)))
          (unwind-protect
               (let ((actual (funcall jf lhs rhs)))
                 (is (equal (array-dimensions out) (array-dimensions actual)) "~A: 形が違う" name)
                 (is (allclose actual out :dtype :f32 :rtol 1e-4 :atol 1e-5) "~A: JAX と一致しない" name))
            (nb::%jit-cache-forget f)))))))

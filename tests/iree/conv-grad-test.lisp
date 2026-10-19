;;;; convolution の jvp / transpose / バッチ化ルール（issue #203）の medium テスト。
;;;;
;;;; (1) jit した grad（lhs と rhs の勾配。転置ルールが出す rev・transpose・reshape・
;;;; batch_group_count 付きの convolution を含む StableHLO）を IREE (local) で実行し、JAX の
;;;; jax.vjp の結果（フィクスチャ conv-grad）と比べる。(2) jit した vmap（バッチ軸を feature group /
;;;; batch group に畳んだ convolution）が、jit しない vmap と一致する。small テスト
;;;; （tests/ad/conv-ad-test.lisp）は同じルールを eager で確かめている。

(in-package #:nabla.iree.tests)

(defun %conv-grad-values-function (params)
  "(x w ct) → L = sum(conv(x, w) * ct) の x と w についての勾配を多値で返す TRACEABLE-FUNCTION。"
  (let* ((loss (nb::%make-traceable-function
                '(x w ct)
                (lambda (x w ct)
                  (nb:reduce-sum
                   (nb::%trace-eqn :mul (list (apply #'nb::%trace-eqn :convolution (list x w) params)
                                              ct))))))
         (g (nb:grad loss :argnums '(0 1))))
    (nb::%make-traceable-function '(x w ct) (lambda (&rest args) (values-list (apply g args))))))

(define-iree-test convolution/jit-grad-matches-jax-fixture
    "jit した convolution の grad（x と w の勾配）を IREE (local) で実行した結果は、JAX の jax.vjp の結果
（stride・padding（負も）・膨張・グループ・depthwise・1〜3 次元・batch-group-count 2）と f32 の許容誤差で一致する。
許容誤差は 1e-4 / 1e-5（総和の順序が JAX と違うため）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil))
    (dolist (case (read-conv-fixture "conv-grad"))
      (destructuring-bind (&key name strides padding lhs-dilation rhs-dilation groups batch-groups
                             lhs rhs ct dlhs drhs &allow-other-keys)
          case
        (let* ((f (%conv-grad-values-function
                   (list :strides strides :padding padding :lhs-dilation lhs-dilation
                         :rhs-dilation rhs-dilation :feature-group-count groups
                         :batch-group-count batch-groups)))
               (jf (nb:jit f :backend backend)))
          (unwind-protect
               (multiple-value-bind (actual-lhs actual-rhs) (funcall jf lhs rhs ct)
                 (is (equal (array-dimensions dlhs) (array-dimensions actual-lhs)) "~A: dlhs の形が違う" name)
                 (is (equal (array-dimensions drhs) (array-dimensions actual-rhs)) "~A: drhs の形が違う" name)
                 (is (allclose actual-lhs dlhs :dtype :f32 :rtol 1e-4 :atol 1e-5) "~A: dlhs が JAX と一致しない" name)
                 (is (allclose actual-rhs drhs :dtype :f32 :rtol 1e-4 :atol 1e-5) "~A: drhs が JAX と一致しない" name))
            (nb::%jit-cache-forget f)))))
    (gc-and-run-finalizers)))

(defparameter *conv-vmap-configs*
  ;; (lhs-shape rhs-shape strides padding lhs-dilation rhs-dilation fgc bgc)
  '(((2 5 5 3) (3 3 3 4) (1 1) ((1 1) (1 1)) (1 1) (1 1) 1 1)
    ((2 5 5 4) (2 2 2 6) (1 1) ((0 0) (0 0)) (1 1) (1 1) 2 1)
    ((4 5 3) (2 3 6) (2) ((1 0)) (1) (1) 1 2))
  "jit した vmap を確かめる設定: 通常、feature group、batch group。")

(define-iree-test convolution/jit-vmap-matches-eager
    "設定（通常・feature group・batch group）× バッチの仕方（lhs だけ・rhs だけ・両方）ごとに、
jit した vmap(convolution) は、jit しない vmap と f32 の許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil))
    (dolist (config *conv-vmap-configs*)
      (destructuring-bind (lhs-shape rhs-shape strides padding ld rd fgc bgc) config
        (let ((f (primitive-function :convolution
                                     (list :strides strides :padding padding :lhs-dilation ld
                                           :rhs-dilation rd :feature-group-count fgc
                                           :batch-group-count bgc)
                                     2)))
          (dolist (in-axes '((0 nil) (nil 1) (1 0)))
            (let* ((g (nb:vmap f :in-axes in-axes :out-axes 0))
                   (jg (nb:jit g :backend backend))
                   (x (make-random-array (make-array-spec (if (first in-axes)
                                                              (axis-at lhs-shape 3 (first in-axes))
                                                              lhs-shape)
                                                          :f32)
                                         :seed 1))
                   (w (make-random-array (make-array-spec (if (second in-axes)
                                                              (axis-at rhs-shape 3 (second in-axes))
                                                              rhs-shape)
                                                          :f32)
                                         :seed 2)))
              (unwind-protect
                   (is (allclose (funcall jg x w) (funcall g x w) :dtype :f32)
                       "設定 ~S in-axes ~S で jit した vmap が eager と一致しない" config in-axes)
                (nb::%jit-cache-forget (nb::%jitted-function-fn jg))))))))
    (gc-and-run-finalizers)))

;;;; conv レイヤーを IREE の local で jit する（issue #207、medium）。
;;;;
;;;; 守らせる性質:
;;;; - jit した順伝播と jit した grad が、Flax のフィクスチャ（conv-test.lisp と同じケース）と一致する。
;;;; - init で作った conv を含む sequential（conv → relu → conv）を jit して、eager と一致する
;;;;   （出力も、モデルと同じ構造の勾配も）。
;;;; - vmap した conv を jit して、eager の vmap と一致する。
;;;; - bf16 の入力・kernel でも jit できる（IREE 3.11 の convert の折り畳みのバグの回避策が効く。
;;;;   CLAUDE.md）。f32 の結果と bf16 の丸めの範囲で一致する。

(in-package #:nabla.nn.tests)

(in-suite :nabla.medium)

(defun %conv-require-iree-or-skip ()
  (or (nabla.iree:iree-available-p :library :both)
      (progn (if (uiop:getenv "NABLA_REQUIRE_IREE")
                 (fail "IREE が見つからない（NABLA_REQUIRE_IREE が設定されている）")
                 (skip "IREE が見つからない"))
             nil)))

(test conv/jit-forward-and-grads-match-flax-on-iree-local
  "conv-test.lisp のフィクスチャの全ケースで、IREE の local で jit した順伝播と jit した grad が
Flax の出力・勾配と一致する。"
  (when (%conv-require-iree-or-skip)
    (let ((backend (nabla:find-backend :iree))
          (nabla:*compile-cache-directory* nil)
          (forward (nabla:with-tracing (m x) (call m x))))
      (dolist (name (%conv-case-names))
        (let* ((case (%cfx-case name))
               (model (%cfx-model case))
               (x (%fx-array case "x"))
               (ct (%fx-array case "ct"))
               (loss (nabla:with-tracing (m x ct) (nabla:reduce-sum (* (funcall forward m x) ct))))
               (jf (nabla:jit forward :backend backend))
               (jg (nabla:jit (nabla:grad loss :argnums '(0 1)) :backend backend)))
          (unwind-protect
               (let* ((grads (funcall jg model x ct))
                      (actual (append (nabla:tree-leaves (first grads)) (list (second grads))))
                      (grad-names (append '("kernel") (when (%fx-entry case "bias") '("bias")) '("x")))
                      (expected (mapcar (lambda (label) (%fx-array case (concatenate 'string "grad." label)))
                                        grad-names)))
                 (is (apply #'%close-f32 (funcall jf model x) (%fx-array case "out") *conv-tolerance*)
                     "~A: jit した順伝播が Flax と一致しない" name)
                 (is (= (length expected) (length actual)) "~A: 勾配の個数" name)
                 (loop for a in actual for e in expected for label in grad-names
                       do (is (apply #'%close-f32 a e *conv-tolerance*)
                              "~A: jit した ~A の勾配が Flax と一致しない" name label)))
            (nabla::%jit-cache-forget forward)
            (nabla::%jit-cache-forget loss)))))))

(test conv/jit-of-an-initialized-conv-net-matches-eager
  "init-conv で作った conv (SAME, stride 2) → relu → conv (VALID) の sequential を jit した出力と、
jit した勾配（モデルと同じ構造）が eager と一致する。"
  (when (%conv-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (net (init-sequential (nabla:prng-key 4)
                                 (list (lambda (k) (init-conv k 3 6 '(3 3) :strides 2))
                                       (lambda (k) (init-activation k :relu))
                                       (lambda (k) (init-conv k 6 4 '(2 2) :padding :valid)))))
           (x (%rand '(2 8 8 3) 8))
           (f (nabla:with-tracing (m x) (call m x)))
           (loss (nabla:with-tracing (m x) (nabla:reduce-sum (call m x))))
           (g (nabla:grad loss))
           (jf (nabla:jit f :backend backend))
           (jg (nabla:jit g :backend backend)))
      (unwind-protect
           (let ((jit-grad (funcall jg net x)) (eager-grad (funcall g net x)))
             (is (apply #'%close-f32 (funcall jf net x) (funcall f net x) *conv-tolerance*))
             (is (typep jit-grad 'sequential))
             (is (every (lambda (a e) (apply #'%close-f32 a e *conv-tolerance*))
                        (nabla:tree-leaves jit-grad) (nabla:tree-leaves eager-grad))))
        (nabla::%jit-cache-forget f)
        (nabla::%jit-cache-forget g)))))

(test conv/jit-of-a-conv-without-bias-matches-eager
  "init-conv :use-bias nil の conv → relu → conv を jit した出力と勾配（葉は kernel の 2 つだけ）が eager と一致する。"
  (when (%conv-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (net (init-sequential (nabla:prng-key 4)
                                 (list (lambda (k) (init-conv k 3 6 '(3 3) :strides 2 :use-bias nil))
                                       (lambda (k) (init-activation k :relu))
                                       (lambda (k) (init-conv k 6 4 '(2 2) :padding :valid :use-bias nil)))))
           (x (%rand '(2 8 8 3) 8))
           (f (nabla:with-tracing (m x) (call m x)))
           (loss (nabla:with-tracing (m x) (nabla:reduce-sum (call m x))))
           (g (nabla:grad loss))
           (jf (nabla:jit f :backend backend))
           (jg (nabla:jit g :backend backend)))
      (unwind-protect
           (let ((jit-grad (funcall jg net x)) (eager-grad (funcall g net x)))
             (is (apply #'%close-f32 (funcall jf net x) (funcall f net x) *conv-tolerance*))
             (is (= 2 (length (nabla:tree-leaves jit-grad))))
             (is (every (lambda (a e) (apply #'%close-f32 a e *conv-tolerance*))
                        (nabla:tree-leaves jit-grad) (nabla:tree-leaves eager-grad))))
        (nabla::%jit-cache-forget f)
        (nabla::%jit-cache-forget g)))))

(test conv/jit-of-vmap-matches-eager-vmap
  "画像ごとに call する関数を vmap し、さらに jit した出力が、eager の vmap と一致する（グループ畳み込み、
SAME、stride 2）。"
  (when (%conv-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (module (make-conv :use-bias t :kernel (%rand '(3 3 2 4) 1) :bias (%rand '(4) 2) :strides '(2 2)
                                :padding :same :kernel-dilation '(1 1) :feature-group-count 2))
           (images (%rand '(3 6 5 4) 3))
           (f (nabla:vmap (nabla:with-tracing (x) (call module x))))
           (jf (nabla:jit (nabla:with-tracing (x) (funcall f x)) :backend backend)))
      (unwind-protect
           (is (apply #'%close-f32 (funcall jf images) (funcall f images) *conv-tolerance*))
        (nabla::%jit-cache-forget jf)))))

(test conv/bf16-forward-and-grad-work-under-jit
  "bf16 の入力・kernel・bias の conv を IREE の local で jit できる（順伝播と、kernel・入力への勾配）。
f32 の eager の結果と bf16 の丸め（rtol 5e-2）の範囲で一致する。"
  (when (%conv-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (kernel (%rand '(3 3 3 4) 1))
           (bias (%rand '(4) 2))
           (x (%rand '(2 6 6 3) 3))
           (ct (%rand '(2 6 6 4) 4))
           (f32 (make-conv :use-bias t :kernel kernel :bias bias :strides '(1 1) :padding :same
                             :kernel-dilation '(1 1) :feature-group-count 1))
           (forward (nabla:with-tracing (k b x)
                      (call (make-conv :use-bias t :kernel k :bias b :strides '(1 1) :padding :same
                                         :kernel-dilation '(1 1) :feature-group-count 1)
                            x)))
           (loss (nabla:with-tracing (k b x ct) (nabla:reduce-sum (* (funcall forward k b x) ct))))
           (jf (nabla:jit forward :backend backend))
           (jg (nabla:jit (nabla:grad loss :argnums '(0 2)) :backend backend))
           (device-args (mapcar (lambda (a) (nabla:to-device (nabla::encode-float16-array a :bf16) backend :dtype :bf16))
                                (list kernel bias x ct))))
      (unwind-protect
           (destructuring-bind (dk db dx dct) device-args
             (let ((out (nabla::decode-float16-array (funcall jf dk db dx) :bf16))
                   (grads (funcall jg dk db dx dct)))
               (is (allclose out (call f32 x) :dtype :f32 :rtol 5d-2 :atol 5d-2))
               (let ((eager (funcall (nabla:grad loss :argnums '(0 2)) kernel bias x ct)))
                 (loop for g in grads for e in eager
                       do (is (allclose (nabla::decode-float16-array g :bf16) e :dtype :f32 :rtol 5d-2 :atol 2d-1))))))
        (mapc #'nabla.iree:release-device-array device-args)
        (nabla::%jit-cache-forget forward)
        (nabla::%jit-cache-forget loss)))))

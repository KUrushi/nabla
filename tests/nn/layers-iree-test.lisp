;;;; 基本レイヤーを IREE の local で jit する（issue #204、medium）。
;;;;
;;;; 守らせる性質:
;;;; - jit した順伝播と jit した grad が、Flax のフィクスチャ（layers-test.lisp と同じケース）と一致する。
;;;; - dropout は jit しても eager と同じ出力（同じキー → 同じマスク）。deterministic は静的引数。
;;;; - init で作ったモデル全体（sequential）を jit して、eager と一致する。

(in-package #:nabla.nn.tests)

(in-suite :nabla.medium)

(defun %layers-require-iree-or-skip ()
  (or (nabla.iree:iree-available-p :library :both)
      (progn (if (uiop:getenv "NABLA_REQUIRE_IREE")
                 (fail "IREE が見つからない（NABLA_REQUIRE_IREE が設定されている）")
                 (skip "IREE が見つからない"))
             nil)))

(test layers/jit-forward-and-grads-match-flax-on-iree-local
  "layers-test.lisp のフィクスチャの全ケース（issue #241 の追加分を含む）で、IREE の local で jit した順伝播と jit した grad が
Flax の出力・勾配と一致する（許容誤差は eager と同じ）。"
  (when (%layers-require-iree-or-skip)
    (let ((backend (nabla:find-backend :iree))
          (nabla:*compile-cache-directory* nil))
      (dolist (entry (append *layers-case-table* *layers-extra-case-table*))
        (destructuring-bind (name make forward ints) entry
          (let* ((case (%fx-case name))
                 (model (funcall make case))
                 (x (%fx-array case "x" :ints ints))
                 (ct (%fx-array case "ct"))
                 (x-grad-p (%fx-entry case "grad.x"))
                 (loss (nabla:with-tracing (m x ct) (nabla:reduce-sum (* (funcall forward m x) ct))))
                 (argnums (append (when model '(0)) (when x-grad-p '(1))))
                 (jf (nabla:jit forward :backend backend))
                 (jg (nabla:jit (nabla:grad loss :argnums argnums) :backend backend)))
            (unwind-protect
                 (let ((grads (funcall jg model x ct))
                       (expected (append (mapcar (lambda (p) (%fx-array case (concatenate 'string "grad." p)))
                                                 (%fx-param-names case))
                                         (when x-grad-p (list (%fx-array case "grad.x"))))))
                   (is (apply #'%close-f32 (funcall jf model x) (%fx-array case "out") (%case-tolerance name))
                       "~A: jit した順伝播が Flax と一致しない" name)
                   (let ((actual (append (when model (nabla:tree-leaves (first grads)))
                                         (when x-grad-p (last grads)))))
                     (is (= (length expected) (length actual)) "~A: 勾配の個数" name)
                     (loop for a in actual for e in expected for i from 0
                           do (is (apply #'%close-f32 a e (%case-tolerance name))
                                  "~A: jit した ~D 番目の勾配が Flax と一致しない" name i))))
              (nabla::%jit-cache-forget forward)
              (nabla::%jit-cache-forget loss))))))))

(test layers/dropout-jit-matches-eager-and-deterministic-is-static
  "dropout を jit した出力は eager と同じ（同じキーなら同じマスク）。deterministic を静的引数にすると、
真のとき入力そのもの（値が一致）、偽のとき dropout がかかる。"
  (when (%layers-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (module (init-dropout (nabla:prng-key 0) 0.3f0))
           (f (nabla:with-tracing (m x key deterministic) (call m x key deterministic)))
           (jf (nabla:jit f :backend backend :static-args '(3)))
           (x (%rand '(16 8) 5))
           (key (nabla:prng-key 77)))
      (unwind-protect
           (progn
             (is (%close-f32 (funcall jf module x key nil) (call module x key nil)))
             (is (not (equalp (funcall jf module x key nil) x)))
             (is (%close-f32 (funcall jf module x key t) x)))
        (nabla::%jit-cache-forget f)))))

(test layers/jit-of-an-initialized-sequential-matches-eager
  "init-sequential で作った dense + gelu + layer-norm + dense のモデルを jit した出力と、jit した
勾配（モデルと同じ構造）が eager と一致する。"
  (when (%layers-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (net (init-sequential (nabla:prng-key 3)
                                 (list (lambda (k) (init-dense k 6 8))
                                       (lambda (k) (init-activation k :gelu))
                                       (lambda (k) (init-layer-norm k 8))
                                       (lambda (k) (init-dense k 8 3)))))
           (x (%rand '(4 6) 8))
           (f (nabla:with-tracing (m x) (call m x)))
           (loss (nabla:with-tracing (m x) (nabla:reduce-sum (call m x))))
           (g (nabla:grad loss))
           (jf (nabla:jit f :backend backend))
           (jg (nabla:jit g :backend backend)))
      (unwind-protect
           (let ((jit-grad (funcall jg net x)) (eager-grad (funcall g net x)))
             (is (%close-f32 (funcall jf net x) (funcall f net x)))
             (is (typep jit-grad 'sequential))
             (is (every #'%close-f32 (nabla:tree-leaves jit-grad) (nabla:tree-leaves eager-grad))))
        (nabla::%jit-cache-forget f)
        (nabla::%jit-cache-forget g)))))

(test layers/dropout-bf16-input-works-under-jit-and-keeps-the-f32-mask
  "bf16 の入力の dropout を IREE の local で jit できる（乱数は f32 で引くので prng-error にならない）。落ちる位置は
同じキーの f32 の eager と同じで、保持した要素は 1/(1-rate) 倍（rate = 0.5 なので 2、bf16 で正確）。"
  (when (%layers-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (module (make-dropout :rate 0.5f0))
           (key (nabla:prng-key 11))
           (ones (%ones '(8 16)))
           (f (nabla:with-tracing (x k) (call module x k nil)))
           (jf (nabla:jit f :backend backend))
           (device-x (nabla:to-device (nabla::encode-float16-array ones :bf16) backend :dtype :bf16)))
      (unwind-protect
           (let ((out (nabla::decode-float16-array (funcall jf device-x key) :bf16))
                 (expected (call module ones key nil)))
             (is (equal (mapcar #'zerop (%elements out)) (mapcar #'zerop (%elements expected))))
             (is (every (lambda (v) (or (zerop v) (= v 2.0))) (%elements out)))
             (is (notevery #'zerop (%elements out))))
        (nabla.iree:release-device-array device-x)
        (nabla::%jit-cache-forget f)))))

(test layers/jit-of-a-sequential-without-bias-matches-eager
  "bias の無い dense（init-dense :use-bias nil）を並べた sequential を jit した出力と勾配（葉は kernel だけ）が
eager と一致する。"
  (when (%layers-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (net (init-sequential (nabla:prng-key 3)
                                 (list (lambda (k) (init-dense k 6 8 :use-bias nil))
                                       (lambda (k) (init-activation k :gelu))
                                       (lambda (k) (init-dense k 8 3 :use-bias nil)))))
           (x (%rand '(4 6) 8))
           (f (nabla:with-tracing (m x) (call m x)))
           (loss (nabla:with-tracing (m x) (nabla:reduce-sum (call m x))))
           (g (nabla:grad loss))
           (jf (nabla:jit f :backend backend))
           (jg (nabla:jit g :backend backend)))
      (unwind-protect
           (let ((jit-grad (funcall jg net x)) (eager-grad (funcall g net x)))
             (is (%close-f32 (funcall jf net x) (funcall f net x)))
             (is (= 2 (length (nabla:tree-leaves jit-grad))))
             (is (every #'%close-f32 (nabla:tree-leaves jit-grad) (nabla:tree-leaves eager-grad))))
        (nabla::%jit-cache-forget f)
        (nabla::%jit-cache-forget g)))))

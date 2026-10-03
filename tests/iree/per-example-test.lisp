;;;; per-example 勾配の IREE 経由 end-to-end テストと JAX フィクスチャ（issue #138、フェーズ3の完了条件）。
;;;;
;;;; 対象は examples/mlp.lisp の MAKE-PER-EXAMPLE-GRAD（vmap (grad 1サンプルの損失)、
;;;; パラメータは in-axes nil、x と y は軸 0）。examples/mlp.lisp の load と f32 配列の読み込みは
;;;; tests/iree/mlp-train-test.lisp の %MLP-EXAMPLE-FN / %TRAIN-FIXTURE-ARRAY を使う。
;;;;
;;;; JAX フィクスチャ: tests/fixtures/per-example/mlp-per-example.lisp
;;;; （生成: tests/fixtures/per-example/generate.py。jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))。
;;;; jax のバージョンと x64 の有無はフィクスチャの :jax-version / :x64 に記録している）。
;;;; 合成（jit (vmap (grad f))）・（jit (grad (sum (vmap f))））・（jit (vmap (vmap g)））の期待値は、
;;;; jit しない eager の結果（性質は tests/per-example-test.lisp の small が確かめている）。

(in-package #:nabla.iree.tests)

(defun %per-example-fixture ()
  (with-open-file (stream (asdf:system-relative-pathname "nabla" "tests/fixtures/per-example/mlp-per-example.lisp"))
    (read stream)))

(defun %per-example-inputs (fixture)
  "フィクスチャの :inputs（x y w1 b1 w2 b2 の順）を (w1 b1 w2 b2 x y) の f32 配列のリストにする。"
  (destructuring-bind (x y &rest params) (mapcar #'%train-fixture-array (getf fixture :inputs))
    (append params (list x y))))

(defun %per-example-check-grads (actual fixture what)
  "ACTUAL（勾配の配列のリスト）が、フィクスチャの :grads と f32 の許容誤差で一致するか。"
  (is (= (length actual) (length (getf fixture :grads))))
  (loop for a in actual
        for entry in (getf fixture :grads)
        do (is (allclose a (%train-fixture-array entry) :dtype :f32)
               "~A: ~A の per-example 勾配が JAX と一致しない" what (first entry))))

(defun %call-values-list (f arrays)
  (multiple-value-list (apply f arrays)))

(define-iree-test per-example/matches-jax-fixture-eager-and-iree
    "examples/mlp.lisp の per-example 勾配（vmap (grad loss)）が、eager でも (jit ...) して
IREE の local で実行しても、JAX の jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)) のフィクスチャと
f32 の許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nb:find-backend :iree))
         (fixture (%per-example-fixture))
         (arrays (%per-example-inputs fixture))
         (per-example (funcall (%mlp-example-fn "MAKE-PER-EXAMPLE-GRAD")))
         (jitted (nb:jit per-example :backend backend)))
    (%per-example-check-grads (%call-values-list per-example arrays) fixture "eager")
    (unwind-protect
         (%per-example-check-grads (%call-values-list jitted arrays) fixture "iree")
      (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))
    (gc-and-run-finalizers)))

(defun %per-example-loss-fn ()
  (funcall (%mlp-example-fn "MAKE-MLP-EXAMPLE-LOSS") 2))

(defun %per-example-grad-of-vmapped-sum ()
  "(w1 b1 w2 b2 x y) → サンプルごとの損失の和の、全パラメータに対する勾配（多値）。"
  (let* ((vmapped (nb:vmap (%per-example-loss-fn) :in-axes '(nil nil nil nil 0 0)))
         (g (nb:grad (nb:with-tracing (w1 b1 w2 b2 x y)
                       (nb:reduce-sum (funcall vmapped w1 b1 w2 b2 x y)))
                     :argnums '(0 1 2 3))))
    (nb:with-tracing (w1 b1 w2 b2 x y)
      (values-list (funcall g w1 b1 w2 b2 x y)))))

(defun %per-example-vmap-of-vmap ()
  "(w1 b1 w2 b2 x y)（x は (M N D)、y は (M N C)）→ per-example 勾配（形は (M N ...)）。"
  (let ((inner (funcall (%mlp-example-fn "MAKE-PER-EXAMPLE-GRAD"))))
    (nb:vmap inner :in-axes '(nil nil nil nil 0 0))))

(defun %per-example-stack-batches (array copies)
  "ARRAY（形 (N ...)）を COPIES 回、新しい先頭軸に積んだ配列（形 (COPIES N ...)）。"
  (let* ((dims (array-dimensions array))
         (out (make-array (cons copies dims) :element-type (array-element-type array)))
         (size (array-total-size array)))
    (dotimes (i (array-total-size out) out)
      (setf (row-major-aref out i) (row-major-aref array (mod i size))))))

(define-iree-test per-example/compositions-match-eager-on-iree
    "jit (grad (sum (vmap f)))、jit (vmap (vmap (grad f)))、内側に :backend つきの jit を持つ
jit の IREE の結果が、jit しない eager の結果と f32 の許容誤差で一致する
（vmap は jit した関数の :backend を見ない。grad と同じ）。"
  (skip-unless-iree :library :both)
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nb:find-backend :iree))
         (fixture (%per-example-fixture))
         (arrays (%per-example-inputs fixture))
         (jitted-fns '()))
    (flet ((check (what f stacked-arrays &optional (reference f))
             (let ((jitted (nb:jit f :backend backend)))
               (push jitted jitted-fns)
               (let ((expected (%call-values-list reference stacked-arrays))
                     (actual (%call-values-list jitted stacked-arrays)))
                 (is (= (length expected) (length actual)))
                 (loop for e in expected for a in actual for i from 0
                       do (is (allclose a e :dtype :f32) "~A: 出力 ~D が eager と一致しない" what i))))))
      (unwind-protect
           (progn
             (check "jit (grad (sum (vmap f)))" (%per-example-grad-of-vmapped-sum) arrays)
             (let ((stacked (append (subseq arrays 0 4)
                                    (mapcar (lambda (a) (%per-example-stack-batches a 3))
                                            (subseq arrays 4)))))
               (check "jit (vmap (vmap (grad f)))" (%per-example-vmap-of-vmap) stacked))
             ;; 内側の jit の :backend は無視され、外側の jit の backend で動く
             (let* ((inner-jit (nb:jit (funcall (%mlp-example-fn "MAKE-PER-EXAMPLE-GRAD"))
                                       :backend :no-such-backend))
                    (outer (nb:with-tracing (w1 b1 w2 b2 x y)
                             (funcall inner-jit w1 b1 w2 b2 x y))))
               ;; eager で outer を直接呼ぶと内側の jit が本当に動くので、期待値は jit の無い関数で作る
               (check "jit (jit :backend ...) の入れ子" outer arrays
                      (funcall (%mlp-example-fn "MAKE-PER-EXAMPLE-GRAD")))))
        (dolist (jitted jitted-fns)
          (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))))
    (gc-and-run-finalizers)))

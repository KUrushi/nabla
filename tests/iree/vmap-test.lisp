;;;; (jit (vmap f)) の IREE 経由 end-to-end テスト（issue #125）。
;;;;
;;;; 期待値は jit しない eager の vmap（性質は tests/vmap-test.lisp の small が
;;;; 参照実装で確かめている）。バッチされていない引数と out-axes の組を1つ通す。

(in-package #:nabla.iree.tests)

(define-iree-test vmap/jit-matches-eager
    "(jit (vmap f)) の結果が、jit しない (vmap f) と f32 の許容誤差で一致する。
f は add と broadcast-in-dim を通り、y はバッチされず、出力は軸 1 に積む。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (f (nb:with-tracing (x y)
              (+ x (nb:broadcast-in-dim y '(3) '(0)))))
         (g (nb:vmap f :in-axes '(1 nil) :out-axes 1))
         (jitted (nb:jit g :backend backend))
         (x (make-random-array (make-array-spec '(3 4) :f32) :seed 1))
         (y (make-random-array (make-array-spec '(1) :f32) :seed 2)))
    (unwind-protect
         (is (allclose (funcall jitted x y) (funcall g x y) :dtype :f32))
      (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))
    (gc-and-run-finalizers)))

;;;; 要素演算のバッチ化ルールの IREE end-to-end テスト（issue #128）。
;;;;
;;;; 14 個の要素演算をすべて通る f を、軸の位置が違う引数とバッチされない引数を
;;;; 混ぜて vmap し、jit（IREE）の結果が jit しない vmap（eager）と一致することを確かめる。
;;;; 性質そのもの（参照実装との一致）は tests/vmap-elementwise-test.lisp の small が見る。

(in-package #:nabla.iree.tests)

(define-iree-test vmap/elementwise-jit-matches-eager
    "(jit (vmap f)) が、jit しない (vmap f) と f32 の許容誤差で一致する。f は
add sub mul div neg exp log tanh max min compare select convert stop-gradient を通り、
x は軸 1、y は軸 0 にバッチされ、z はバッチされない。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (f (nb:with-tracing (x y z)
              (let* ((a (+ (- x y) (* z (/ x (exp y)))))
                     (b (max (- a) (min (tanh a) (log (exp z)))))
                     (c (nb:where (< x y) b (nb:stop-gradient a))))
                (nb:convert (nb:convert c :f64) :f32))))
         (g (nb:vmap f :in-axes '(1 0 nil) :out-axes 1))
         (jitted (nb:jit g :backend backend))
         (x (make-random-array (make-array-spec '(3 4) :f32) :seed 1))
         (y (make-random-array (make-array-spec '(4 3) :f32) :seed 2))
         (z (make-random-array (make-array-spec '(3) :f32) :seed 3)))
    (unwind-protect
         (is (allclose (funcall jitted x y z) (funcall g x y z) :dtype :f32))
      (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))
    (gc-and-run-finalizers)))

;;;; jit / grad の PyTree の引数と戻り値の IREE 経由 end-to-end テスト（issue #199、medium）。
;;;;
;;;; 守らせる性質:
;;;; - jit(f)(tree) は eager の f(tree) と一致し、戻り値の木の形（treedef）は eager と同じ。
;;;; - (jit (grad f)) / (jit (value-and-grad f)) の結果は、jit しない grad の結果と同じ
;;;;   treedef で、葉ごとに一致する。
;;;; - 葉が host の配列でも device-array でも同じ結果になる。
;;;; IREE はコンパイルが遅いので、PBT の試行回数は小さく抑える。

(in-package #:nabla.iree.tests)

(defun %leaves-close-p (actual expected dtype)
  (and (nb:treedef= (nb:tree-structure actual) (nb:tree-structure expected))
       (every (lambda (a e) (allclose a e :dtype dtype))
              (nb:tree-leaves actual) (nb:tree-leaves expected))))

(define-iree-test pytree-function/jit-matches-eager-on-random-trees
    "ランダムな木（リスト・ベクタ・登録した型）の f32 の葉に tanh(x+x) を適用する関数を IREE で jit
した結果は、eager の結果と同じ treedef で、葉ごとに f32 の許容誤差内で一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 10)
        (nb:*compile-cache-directory* nil))
    (is (check-it (generator (tuple (pytree-spec)))
                  (lambda (value)
                    (let ((tree (pytree-array-tree (first value) :f32 :max-leaves 8))
                          (f (pytree-map-function)))
                      ;; 葉の無い木は入出力が空の関数になるので対象外
                      (or (null (nb:tree-leaves tree))
                          (let ((jf (nb:jit f :backend backend)))
                            (unwind-protect
                                 (%leaves-close-p (funcall jf tree) (funcall f tree) :f32)
                              (nb::%jit-cache-forget f))))))
                  :regression-id pytree-function/jit-matches-eager-on-random-trees
                  :regression-file (regression-path "iree-pytree-function-jit-matches-eager"
                                                    :package "NABLA.IREE.TESTS")))
    (gc-and-run-finalizers)))

(define-iree-test pytree-function/jit-grad-matches-eager-grad-on-random-trees
    "(jit (value-and-grad loss)) と (jit (grad loss)) の木を受けて木を返す結果は、jit しない
grad と同じ treedef で、葉ごとに f32 の許容誤差内で一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 6)
        (nb:*compile-cache-directory* nil))
    (is (check-it (generator (tuple (pytree-spec)))
                  (lambda (value)
                    (let* ((tree (pytree-array-tree (first value) :f32 :max-leaves 8))
                           (loss (pytree-loss-function))
                           (g (nb:grad loss))
                           (vg (nb:value-and-grad loss)))
                      (or (null (nb:tree-leaves tree))
                          (let ((jg (nb:jit g :backend backend))
                                (jvg (nb:jit vg :backend backend)))
                            (unwind-protect
                                 (and (%leaves-close-p (funcall jg tree) (funcall g tree) :f32)
                                      (multiple-value-bind (value grads) (funcall jvg tree)
                                        (multiple-value-bind (e-value e-grads) (funcall vg tree)
                                          (and (allclose value e-value :dtype :f32)
                                               (%leaves-close-p grads e-grads :f32)))))
                              (nb::%jit-cache-forget g)
                              (nb::%jit-cache-forget vg))))))
                  :regression-id pytree-function/jit-grad-matches-eager-grad-on-random-trees
                  :regression-file (regression-path "iree-pytree-function-jit-grad"
                                                    :package "NABLA.IREE.TESTS")))
    (gc-and-run-finalizers)))

(define-iree-test pytree-function/jit-accepts-device-array-and-number-leaves
    "葉が host の配列・device-array・Lisp の実数のどれでも、jit の結果は同じになる。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (f (pytree-map-function))
         (jf (nb:jit f :backend backend))
         (host (make-random-array (make-array-spec '(3) :f32) :seed 3))
         (device (nb:to-device host backend))
         (scalar (make-random-array (make-array-spec '() :f32) :seed 4)))
    (unwind-protect
         (let ((expected (funcall f (list host scalar))))
           (is (%leaves-close-p (funcall jf (list device scalar)) expected :f32))
           (is (%leaves-close-p (funcall jf (list host (aref scalar))) expected :f32)))
      (nb::%jit-cache-forget f))
    (gc-and-run-finalizers)))

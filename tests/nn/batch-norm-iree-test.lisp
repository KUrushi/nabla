;;;; batch-norm を IREE の local で jit する（issue #260、medium）。
;;;;
;;;; 守らせる性質: jit した call（use-running-average は静的引数）と jit した (value-and-grad loss :has-aux t) が、
;;;; Flax のフィクスチャの全ケースで学習（移動平均の更新を含む数ステップ）と推論の両方で一致する
;;;; （batch-norm-test.lisp の %check-bn-flax-case を jit 版の関数で回す）。jit の戻り値の多値
;;;; (values 出力 新しいモジュール) と aux のモジュールが、元と同じ treedef の木で戻ることも含む。

(in-package #:nabla.nn.tests)

(in-suite :nabla.medium)

(test batch-norm/jit-matches-flax-on-iree-local
  "IREE の local で jit した call と value-and-grad（:has-aux）が、フィクスチャの全ケースで Flax と一致する。"
  (when (%layers-require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (forward (nabla:with-tracing (m x mode) (call m x mode)))
           (jitted-forward (nabla:jit forward :backend backend :static-args '(2)))
           (jitted '()))
      (unwind-protect
           (dolist (name *bn-case-names*)
             (%check-bn-flax-case
              name
              :value-and-grad (lambda (f &rest options)
                                (let ((j (nabla:jit (apply #'nabla:value-and-grad f options) :backend backend)))
                                  (push j jitted)
                                  j))
              :call-function jitted-forward))
        (nabla::%jit-cache-forget forward)
        (dolist (j jitted) (nabla::%jit-cache-forget (nabla::%jitted-function-fn j)))))))

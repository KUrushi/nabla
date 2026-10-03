;;;; 形状演算・縮約・dot-general のバッチ化ルールの IREE 経由 end-to-end テスト（issue #129）。
;;;;
;;;; (jit (vmap f)) を IREE の local でコンパイル・実行した結果が、jit しない (vmap f) の
;;;; eager の結果と f32 の許容誤差で一致する。性質そのものは tests/vmap-shape-test.lisp の
;;;; small が参照実装で確かめている。

(in-package #:nabla.iree.tests)

(defun %shape-rule-function (name params arity)
  "プリミティブ NAME を PARAMS で1回呼ぶ ARITY 引数の TRACEABLE-FUNCTION
（配列なら eager 実装、トレーサなら eqn を足す）。"
  (nb::%make-traceable-function
   (loop for i below arity collect (intern (format nil "X~D" i)))
   (lambda (&rest args)
     (if (some (lambda (a) (typep a 'nb::tracer)) args)
         (apply #'nb::%trace-eqn name args params)
         (apply (nb::primitive-eager (nb::find-primitive name))
                args (mapcar #'nb:array-aval args) params)))))

(defparameter *vmap-shape-cases*
  ;; (名前 params 引数の形 in-axes out-axes)。引数の形はバッチ軸込み。
  '((:transpose (:perm (1 0)) ((3 2 4)) (1) 2)
    (:reshape (:shape (6)) ((2 3 4)) (2) 1)
    (:broadcast-in-dim (:shape (3 2) :dims (1)) ((2 4)) (1) 0)
    (:reduce-sum (:axes (0)) ((3 4 2)) (1) 1)
    (:reduce-max (:axes (0 1)) ((3 2 4)) (2) 0)
    ;; lhs だけ・rhs だけ・両側・既存の batch 次元つき
    (:dot-general (:lhs-contracting (1) :rhs-contracting (0) :lhs-batch () :rhs-batch ())
     ((3 4 2) (2 5)) (1 nil) 1)
    (:dot-general (:lhs-contracting (1) :rhs-contracting (0) :lhs-batch () :rhs-batch ())
     ((3 2) (4 2 5)) (nil 0) 2)
    (:dot-general (:lhs-contracting (2) :rhs-contracting (1) :lhs-batch (0) :rhs-batch (0))
     ((2 3 4 5) (2 5 3 3)) (1 3) 0)))

(define-iree-test vmap-shape/jit-matches-eager
    "形状・縮約・dot-general のバッチ化ルールを通した (jit (vmap f)) の結果が、jit しない
(vmap f) と f32 の許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil))
    (loop for (name params shapes in-axes out-axes) in *vmap-shape-cases*
          for k from 1
          do (let* ((f (%shape-rule-function name params (length shapes)))
                    (g (nb:vmap f :in-axes in-axes :out-axes out-axes))
                    (jitted (nb:jit g :backend backend))
                    (args (loop for shape in shapes for i from 0
                                collect (make-random-array (make-array-spec shape :f32)
                                                           :seed (+ (* 10 k) i)))))
               (unwind-protect
                    (is (allclose (apply jitted args) (apply g args) :dtype :f32)
                        "~S ~S" name params)
                 (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))))
    (gc-and-run-finalizers)))

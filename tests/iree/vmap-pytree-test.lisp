;;;; vmap の PyTree の引数・戻り値・in-axes / out-axes の IREE 経由 end-to-end テスト
;;;; （issue #200、medium）。
;;;;
;;;; 守らせる性質:
;;;; - (jit (vmap f)) は、ランダムな木（リスト・ベクタ・登録した型）の全葉を軸 0 でバッチし、
;;;;   出力の最後の軸へ積み直す設定でも、jit しない vmap と同じ treedef・f32 の許容誤差内で一致する。
;;;; - パラメータの木をバッチせずデータだけをバッチする per-example 勾配
;;;;   (jit (vmap (grad loss) :in-axes '(nil 0 0))) は、jit しない結果と一致し、勾配はパラメータと
;;;;   同じ木になる（defmodule 版は tests/nn/module-vmap-test.lisp）。
;;;; IREE はコンパイルが遅いので、PBT の試行回数は小さく抑える。

(in-package #:nabla.iree.tests)

(defun %vp-close-p (actual expected)
  (and (nb:treedef= (nb:tree-structure actual) (nb:tree-structure expected))
       (every (lambda (a e) (allclose a e :dtype :f32))
              (nb:tree-leaves actual) (nb:tree-leaves expected))))

(defun %vp-batched-tree (spec size)
  "SPEC の葉 n を、先頭に長さ SIZE のバッチ軸を持つ f32 の乱数配列にした木（葉は 8 個まで）。"
  (let ((count 0))
    (nb:tree-map (lambda (n)
                   (when (< (incf count) 9)
                     (make-random-array
                      (make-array-spec (cons size (ecase (mod n 3) (0 '()) (1 '(3)) (2 '(2 3))))
                                       :f32)
                      :seed (+ 2000 n))))
                 spec)))

(define-iree-test vmap-pytree/jit-of-vmap-matches-eager-on-random-trees
    "(jit (vmap f :in-axes '(0) :out-axes '(-1))) は、ランダムな木を受けて木を返す f について、
jit しない vmap と同じ treedef で、葉ごとに f32 の許容誤差内で一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 6)
        (nb:*compile-cache-directory* nil))
    (is (check-it (generator (tuple (pytree-spec)))
                  (lambda (value)
                    (let ((tree (%vp-batched-tree (first value) 2)))
                      (or (null (nb:tree-leaves tree))
                          (let* ((v (nb:vmap (pytree-map-function) :in-axes '(0) :out-axes '(-1)))
                                 (jv (nb:jit v :backend backend)))
                            (unwind-protect
                                 (%vp-close-p (funcall jv tree) (funcall v tree))
                              (nb::%jit-cache-forget v))))))
                  :regression-id vmap-pytree/jit-of-vmap-matches-eager-on-random-trees
                  :regression-file (regression-path "iree-vmap-pytree-jit-matches-eager"
                                                    :package "NABLA.IREE.TESTS")))
    (gc-and-run-finalizers)))

(define-iree-test vmap-pytree/jit-per-example-gradients-of-a-parameter-tree
    "パラメータの木 (w b) をバッチせず x と y だけをバッチした per-example 勾配を jit して IREE の
local で実行した結果は、jit しない結果と一致し、勾配は (w b) と同じ木で、軸 0 にサンプルを積んである。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (term (nb:with-tracing (w b x y) (* (tanh (+ (* w x) b)) y)))
         (loss (nb::%make-traceable-function
                (list (gensym "PARAMS") (gensym "X") (gensym "Y"))
                (lambda (params x y)
                  (destructuring-bind (w b) params
                    (nb:reduce-sum (funcall term w b x y))))))
         (per-example (nb:vmap (nb:grad loss) :in-axes '(nil 0 0)))
         (jp (nb:jit per-example :backend backend))
         (params (list (make-random-array (make-array-spec '(3) :f32) :seed 1)
                       (make-random-array (make-array-spec '(3) :f32) :seed 2)))
         (xs (make-random-array (make-array-spec '(4 3) :f32) :seed 3))
         (ys (make-random-array (make-array-spec '(4 3) :f32) :seed 4)))
    (unwind-protect
         (let ((result (funcall jp params xs ys))
               (expected (funcall per-example params xs ys)))
           (is (%vp-close-p result expected))
           (is (equal '(4 3) (array-dimensions (first result))))
           (is (nb:treedef= (nb:tree-structure params) (nb:tree-structure result))))
      (nb::%jit-cache-forget per-example))
    (gc-and-run-finalizers)))

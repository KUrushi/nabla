;;;; cond* / while-loop のバッチ化ルールの IREE 経由 end-to-end テスト（issue #140）。
;;;;
;;;; (jit (vmap f)) を IREE の local でコンパイル・実行した結果が、jit しない (vmap f) の
;;;; eager の結果と f32 の許容誤差で一致する。性質そのものは tests/vmap-control-test.lisp の
;;;; small が参照実装で確かめている。条件がバッチされる場合（select / any-reduce の形）と、
;;;; されない場合（バッチ化した stablehlo.if / stablehlo.while）の両方を通す。
;;;; while の carry に比較由来の値を持たせない（docs/stablehlo-ops.md の制約）。

(in-package #:nabla.iree.tests)

(defparameter *vc-then*
  (nb:with-tracing (u v) (values (+ u v) (* u 2.0))))

(defparameter *vc-else*
  (nb:with-tracing (u v) (values (- u v) v)))

(defparameter *vc-cond*
  (nb:with-tracing (x y)
    (multiple-value-bind (a b)
        (nb:cond* (< (nb:reduce-sum x :axes '(0)) 0.0) *vc-then* *vc-else* x y)
      (values a b))))

(defparameter *vc-cond-y-pred*
  (nb:with-tracing (x y)
    (multiple-value-bind (a b)
        (nb:cond* (< (nb:reduce-sum y :axes '(0)) 0.0) *vc-then* *vc-else* x y)
      (values a b))))

(defparameter *vc-while-count*
  (nb:with-tracing (x limit)
    (let ((r (nb:while-loop (nb:with-tracing (c) (< (first c) limit))
                            (nb:with-tracing (c) (list (+ (first c) 1.0) (+ (* (second c) 1.5) 1.0)))
                            (list (nb::%scalar-array 0.0 :f32) x))))
      (values (first r) (second r)))))

(defparameter *vc-while-captured*
  (nb:with-tracing (x y)
    (let ((r (nb:while-loop (nb:with-tracing (c) (< (first c) 3.0))
                            (nb:with-tracing (c) (list (+ (first c) 1.0) (+ (second c) x)))
                            (list (nb::%scalar-array 0.0 :f32) y))))
      (values (first r) (second r)))))

(defparameter *vmap-control-cases*
  ;; (名前 関数 引数の形（バッチ軸込み）in-axes)。反復回数 limit は 0〜4 の整数値にする。
  `((cond-batched-pred ,*vc-cond* ((4 3) (3)) (0 nil))
    (cond-batched-pred-both ,*vc-cond* ((3 4) (4 3)) (1 0))
    (cond-unbatched-pred ,*vc-cond-y-pred* ((4 3) (3)) (0 nil))
    (cond-unbatched-pred-y-batched ,*vc-cond-y-pred* ((3) (3 4)) (nil 1))
    (while-batched-pred ,*vc-while-count* ((3) (4)) (nil 0) :limit)
    (while-batched-pred-both ,*vc-while-count* ((4 3) (4)) (0 0) :limit)
    (while-carry-becomes-batched ,*vc-while-captured* ((4 3) (3)) (0 nil))))

(define-iree-test vmap-control/jit-matches-eager
    "cond* / while-loop のバッチ化ルールを通した (jit (vmap f)) の結果が、jit しない (vmap f) と
f32 の許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil))
    (loop for (name f shapes in-axes limit) in *vmap-control-cases*
          for k from 1
          do (let* ((g (nb:vmap f :in-axes in-axes :out-axes 0))
                    (jitted (nb:jit g :backend backend))
                    (args (loop for shape in shapes for i from 0
                                collect (let ((a (make-random-array (make-array-spec shape :f32)
                                                                    :seed (+ (* 10 k) i))))
                                          (when (and limit (= i 1))
                                            (dotimes (j (array-total-size a))
                                              (setf (row-major-aref a j) (float (mod (+ k j) 5) 1.0))))
                                          a))))
               (unwind-protect
                    (let ((actual (multiple-value-list (apply jitted args)))
                          (expected (multiple-value-list (apply g args))))
                      (is (and (= (length actual) (length expected))
                               (every (lambda (a e) (allclose a e :dtype :f32)) actual expected))
                          "~S" name))
                 (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))))
    (gc-and-run-finalizers)))

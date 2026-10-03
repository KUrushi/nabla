;;;; per-example 勾配: vmap と grad・jit の合成の性質（issue #138、eager）。
;;;;
;;;; 対象は examples/mlp.lisp と同じ式の2層 MLP（dense -> tanh -> dense の softmax
;;;; 交差エントロピー）。サンプルごとの損失 %PE-EXAMPLE-LOSS（x: (D)、y: (C)）と、
;;;; バッチ全体の損失 %PE-BATCH-LOSS（x: (N D)、y: (N C)、係数 -1/N）をここで定義する
;;;; （examples/mlp.lisp は load すると学習が走り IREE が要るので、small では load しない）。
;;;; f64 で比べる。IREE を通す合成と JAX フィクスチャは tests/iree/per-example-test.lisp（medium）。
;;;;
;;;; 守らせる性質:
;;;; - per-example 勾配の平均 == バッチ全体の損失の grad
;;;; - (grad (sum (vmap loss))) == per-example 勾配の和 == N * バッチ平均損失の grad
;;;; - (vmap (vmap f)) は、外側の軸で切り出して内側の vmap を適用し積み直したものと一致する
;;;; - vmap が jit した関数をトレース中に呼ぶと、grad と同じく中の関数だけを使い :backend は見ない

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %pe-example-loss (c)
  "1サンプルの損失 (w1 b1 w2 b2 x y) → スカラー（f64）。C はクラス数。"
  (nb:with-tracing (w1 b1 w2 b2 x y)
    (let* ((hidden (tanh (+ (nb:dot x w1) b1)))
           (logits (+ (nb:dot hidden w2) b2))
           (m (nb:stop-gradient (nb:reduce-max logits :axes '(0))))
           (shifted (- logits (nb:broadcast-in-dim m (list c) '())))
           (lse (log (nb:reduce-sum (exp shifted) :axes '(0))))
           (logp (- shifted (nb:broadcast-in-dim lse (list c) '()))))
      (- (nb:reduce-sum (* y logp))))))

(defun %pe-batch-loss (n h c)
  "バッチ全体の損失 (w1 b1 w2 b2 x y) → スカラー（f64）。N はバッチの長さ、H は隠れ層の幅、
C はクラス数。"
  (let ((scale (/ -1d0 n)))
    (nb:with-tracing (w1 b1 w2 b2 x y)
      (let* ((hidden (tanh (+ (nb:dot x w1) (nb:broadcast-in-dim b1 (list n h) '(1)))))
             (logits (+ (nb:dot hidden w2) (nb:broadcast-in-dim b2 (list n c) '(1))))
             (m (nb:stop-gradient (nb:reduce-max logits :axes '(1))))
             (shifted (- logits (nb:broadcast-in-dim m (list n c) '(0))))
             (lse (log (nb:reduce-sum (exp shifted) :axes '(1))))
             (logp (- shifted (nb:broadcast-in-dim lse (list n c) '(0)))))
        (* (nb:reduce-sum (* y logp)) scale)))))

(defun %pe-grads-as-values (loss)
  "LOSS の全パラメータ（引数 0..3）に対する勾配を多値で返す関数。"
  (let ((g (nb:grad loss :argnums '(0 1 2 3))))
    (nb:with-tracing (w1 b1 w2 b2 x y)
      (values-list (funcall g w1 b1 w2 b2 x y)))))

(defparameter *pe-in-axes* '(nil nil nil nil 0 0))

(defun %pe-arrays (d h c lead seed)
  "(w1 b1 w2 b2 x y) の f64 乱数配列のリスト。x は LEAD（バッチ軸の長さのリスト）+ (D)、
y は LEAD + (C)（one-hot ではなく乱数。損失は y について線形なのでどんな y でも性質は変わらない）。"
  (flet ((arr (shape offset)
           (make-random-array (make-array-spec shape :f64) :seed (+ seed offset))))
    (list (arr (list d h) 0) (arr (list h) 1) (arr (list h c) 2) (arr (list c) 3)
          (arr (append lead (list d)) 4) (arr (append lead (list c)) 5))))

(defun %pe-case-generator ()
  (generator (tuple (uniform-integer :lo 1 :hi 4)       ; 0 D
                    (uniform-integer :lo 1 :hi 5)       ; 1 H
                    (uniform-integer :lo 2 :hi 4)       ; 2 C
                    (uniform-integer :lo 1 :hi 5)       ; 3 N
                    (uniform-integer :lo 0 :hi 10000)))) ; 4 seed

(defun %pe-mean-over-axis-0 (array)
  "ARRAY の軸 0 の平均（f64）。参照実装として、足し込みを Lisp で直接書く。"
  (let* ((dims (array-dimensions array))
         (n (first dims))
         (out (make-array (rest dims) :element-type 'double-float :initial-element 0d0))
         (inner (array-total-size out)))
    (dotimes (i (array-total-size array) out)
      (incf (row-major-aref out (mod i inner)) (/ (row-major-aref array i) n)))))

(test per-example/mean-of-per-example-grads-equals-batch-grad
  "per-example 勾配（vmap (grad 1サンプルの損失)、パラメータは in-axes nil）の軸 0 の平均が、
バッチ全体の損失の grad と f64 の許容誤差で一致する。"
  (is (check-it (%pe-case-generator)
                (lambda (case)
                  (destructuring-bind (d h c n seed) case
                    (let* ((arrays (%pe-arrays d h c (list n) seed))
                           (per-example (multiple-value-list
                                         (apply (nb:vmap (%pe-grads-as-values (%pe-example-loss c))
                                                         :in-axes *pe-in-axes*)
                                                arrays)))
                           (batch (multiple-value-list
                                   (apply (%pe-grads-as-values (%pe-batch-loss n h c)) arrays))))
                      (every (lambda (pe b) (allclose (%pe-mean-over-axis-0 pe) b :dtype :f64))
                             per-example batch))))
                :regression-id per-example/mean-equals-batch-grad
                :regression-file (regression-path "per-example-grad"))))

(test per-example/grad-of-vmapped-sum-equals-sum-of-per-example-grads
  "grad の中の vmap: (grad (lambda (p) (sum (vmap loss ...)))) の勾配が、N * バッチ平均損失の
grad（= サンプルごとの勾配の和）と一致する。vmap の中で閉包ではなく引数として x y を渡す。"
  (is (check-it (%pe-case-generator)
                (lambda (case)
                  (destructuring-bind (d h c n seed) case
                    (let* ((arrays (%pe-arrays d h c (list n) seed))
                           (vmapped (nb:vmap (%pe-example-loss c) :in-axes *pe-in-axes*))
                           (summed (nb:with-tracing (w1 b1 w2 b2 x y)
                                     (nb:reduce-sum (funcall vmapped w1 b1 w2 b2 x y))))
                           (actual (multiple-value-list (apply (%pe-grads-as-values summed) arrays)))
                           (batch (multiple-value-list (apply (%pe-grads-as-values (%pe-batch-loss n h c)) arrays))))
                      (every (lambda (a b) (allclose a (map-array (lambda (v) (* v n)) b) :dtype :f64))
                             actual batch))))
                :regression-id per-example/grad-of-vmapped-sum
                :regression-file (regression-path "per-example-grad"))))

(defun map-array (fn array)
  (let ((out (make-array (array-dimensions array) :element-type (array-element-type array))))
    (dotimes (i (array-total-size array) out)
      (setf (row-major-aref out i) (funcall fn (row-major-aref array i))))))

(test per-example/vmap-of-vmap-matches-per-slice-reference
  "vmap の中の vmap: 軸が2つ（M と N）あるバッチの per-example 勾配が、外側の軸で切り出して
内側の vmap を適用し積み直した参照実装（REFERENCE-VMAP）と一致する。"
  (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 3) ; M
                                  (uniform-integer :lo 1 :hi 3) ; N
                                  (uniform-integer :lo 0 :hi 10000)))
                (lambda (case)
                  (destructuring-bind (m n seed) case
                    (let* ((arrays (%pe-arrays 2 3 2 (list m n) seed))
                           (inner (nb:vmap (%pe-grads-as-values (%pe-example-loss 2))
                                           :in-axes *pe-in-axes*))
                           (outer (nb:vmap inner :in-axes *pe-in-axes*))
                           (expected (reference-vmap inner arrays :in-axes *pe-in-axes* :out-axes 0))
                           (actual (multiple-value-list (apply outer arrays))))
                      (and (= (length expected) (length actual))
                           (every (lambda (a e) (allclose a e :dtype :f64)) actual expected)))))
                :regression-id per-example/vmap-of-vmap
                :regression-file (regression-path "per-example-grad"))))

(test per-example/vmap-of-jitted-function-ignores-backend
  "vmap に jit した関数を渡すと、grad と同じく中の関数だけを使い、その :backend は見ない
（存在しない backend 名を渡しても eager の vmap が動き、jit しない関数と同じ結果になる）。"
  (let* ((f (nb:with-tracing (x y) (* (+ x y) x)))
         (jitted (nb:jit f :backend :no-such-backend))
         (x (make-random-array (make-array-spec '(3 2) :f64) :seed 1))
         (y (make-random-array (make-array-spec '(3 2) :f64) :seed 2)))
    (is (allclose (funcall (nb:vmap jitted) x y) (funcall (nb:vmap f) x y) :dtype :f64))
    (is (allclose (funcall (nb:grad (nb:with-tracing (a b) (nb:reduce-sum (funcall jitted a b)))) x y)
                  (funcall (nb:grad (nb:with-tracing (a b) (nb:reduce-sum (funcall f a b)))) x y)
                  :dtype :f64))))

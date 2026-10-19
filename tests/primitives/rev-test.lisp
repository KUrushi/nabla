;;;; rev プリミティブ（軸の反転。issue #203）の small テスト。
;;;;
;;;; rev は convolution の lhs の転置（JAX の _conv_general_dilated_transpose_lhs が
;;;; カーネルの空間軸を rev する）のために足した。形状推論・eager・StableHLO の出力と、
;;;; jvp / transpose / バッチ化ルールを、独立に書いた添字のオラクルと性質で確かめる。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %rev-case (seed)
  "SEED から (shape dims) を作る。rank 1〜3、各次元 1〜4、dims は昇順とは限らない重複のない部分集合（空もある）。"
  (let* ((rank (1+ (mod seed 3)))
         (shape (loop for i below rank collect (1+ (mod (floor seed (expt 5 (1+ i))) 4))))
         (dims (remove-if-not (lambda (a) (logbitp a (floor seed 7))) (loop for i below rank collect i))))
    (list shape (if (oddp (floor seed 11)) (reverse dims) dims))))

(defun %rev-oracle (array dims)
  "ARRAY の軸 DIMS を反転した配列。各要素を添字から直接引く（実装とは別の作り方）。"
  (let* ((shape (array-dimensions array))
         (result (make-array shape :element-type (array-element-type array))))
    (dotimes (i (array-total-size result) result)
      (let* ((subs (let ((rest i) (acc '()))
                     (dolist (d (reverse shape) acc) (push (mod rest d) acc) (setf rest (floor rest d)))))
             (src (loop for s in subs for axis from 0 for d in shape
                        collect (if (member axis dims) (- d 1 s) s))))
        (setf (row-major-aref result i) (apply #'aref array src))))))

(test rev/eager-matches-oracle-for-every-dtype
  "eager の rev は、独立に書いた添字のオラクルと一致する（すべての float dtype と整数 dtype。
値を解釈しない構造演算なので、ビットがそのまま動く）。形状推論は入力と同じ aval を返す。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (destructuring-bind (shape dims) (%rev-case seed)
                    (every (lambda (dtype)
                             (let* ((x (make-random-array (make-array-spec shape dtype) :seed seed))
                                    (aval (nb:array-aval x dtype)))
                               (and (equalp aval (%abstract-eval-of :rev (list aval) :dims dims))
                                    (equalp (%rev-oracle x dims)
                                            (%eager-of :rev (list x) (list aval) :dims dims)))))
                           (append *dtypes* *integer-dtypes*))))
                :regression-id rev/eager-matches-oracle-for-every-dtype
                :regression-file (regression-path "rev-eager"))))

(test rev/is-an-involution
  "rev を同じ軸でもう一度かけると元に戻る。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (destructuring-bind (shape dims) (%rev-case seed)
                    (let* ((x (make-random-array (make-array-spec shape :f64) :seed seed))
                           (aval (nb:array-aval x :f64)))
                      (equalp x (%eager-of :rev (list (%eager-of :rev (list x) (list aval) :dims dims))
                                           (list aval) :dims dims)))))
                :regression-id rev/is-an-involution
                :regression-file (regression-path "rev-involution"))))

(test rev/rejects-invalid-dims
  "dims は重複のない、範囲内の整数のリスト。違えば PRIMITIVE-ERROR。入力の個数も検査する。"
  (let ((aval (nb:make-aval '(2 3) :f32)))
    (is (equalp aval (%abstract-eval-of :rev (list aval) :dims '())))
    (signals nb:primitive-error (%abstract-eval-of :rev (list aval) :dims '(2)))
    (signals nb:primitive-error (%abstract-eval-of :rev (list aval) :dims '(-1)))
    (signals nb:primitive-error (%abstract-eval-of :rev (list aval) :dims '(0 0)))
    (signals nb:primitive-error (%abstract-eval-of :rev (list aval) :dims '(a)))
    (signals nb:primitive-error (%abstract-eval-of :rev (list aval) :dims 0))
    (signals nb:primitive-error (%abstract-eval-of :rev (list aval aval) :dims '(0)))))

(test rev/emit-golden
  "stablehlo.reverse の1行になる。"
  (is (string= "%0 = stablehlo.reverse %a, dims = [0, 2] : tensor<2x3x4xf32>"
               (%emit-of :rev '("%a") (list (nb:make-aval '(2 3 4) :f32)) "%0"
                         (nb:make-aval '(2 3 4) :f32) :dims '(0 2)))))

;;; --- 自動微分とバッチ化 ---

(defun %rev-graph (shape dims)
  (%prim-graph :rev (list (nb:make-aval shape :f64)) :dims dims))

(test rev/jvp-matches-central-difference
  "rev の jvp は f64 の中心差分と一致する（rev は線形なので接線にも rev がかかる）。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (destructuring-bind (shape dims) (%rev-case seed)
                    (%jvp-matches-central-difference-p (%rev-graph shape dims) seed)))
                :regression-id rev/jvp-matches-central-difference
                :regression-file (regression-path "rev-jvp"))))

(test rev/transpose-is-rev
  "rev の転置は同じ軸の rev: <rev(v), u> = <v, T(u)> が成り立つ（vjp の余接線が rev(u) と一致する）。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (destructuring-bind (shape dims) (%rev-case seed)
                    (let* ((aval (nb:make-aval shape :f64))
                           (x (make-random-array (make-array-spec shape :f64) :seed seed))
                           (u (make-random-array (make-array-spec shape :f64) :seed (1+ seed)))
                           (v (make-random-array (make-array-spec shape :f64) :seed (+ 2 seed)))
                           (ct (first (%vjp-cotangents :rev (list aval) (list :dims dims) (list x u)))))
                      (and (allclose ct (%rev-oracle u dims) :dtype :f64)
                           (%tr-close-p (inner-product (%rev-oracle v dims) u) (inner-product v ct) 1d-9)))))
                :regression-id rev/transpose-is-rev
                :regression-file (regression-path "rev-transpose"))))

(test rev/batch-rule-matches-per-slice-reference
  "rev の vmap は、バッチ軸で切り出した各要素に rev をかけて積み直したものと一致する
（バッチ軸の位置・out-axes もランダム）。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 0 :hi 99)
                                  (uniform-integer :lo 0 :hi 99)))
                (lambda (case)
                  (destructuring-bind (seed code-axis code-out) case
                    (destructuring-bind (shape dims) (%rev-case seed)
                      (let* ((f (primitive-function :rev (list :dims dims) 1))
                             (axis (mod code-axis (1+ (length shape))))
                             (out (mod code-out (1+ (length shape))))
                             (x (%vmap-batched-array shape 3 axis seed))
                             (expected (reference-vmap f (list x) :in-axes axis :out-axes out))
                             (actual (funcall (nb:vmap f :in-axes axis :out-axes out) x)))
                        (allclose actual (first expected) :dtype :f64)))))
                :regression-id rev/batch-rule-matches-per-slice-reference
                :regression-file (regression-path "rev-batch"))))

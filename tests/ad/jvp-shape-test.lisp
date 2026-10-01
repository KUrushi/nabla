;;;; 形状・縮約・dot-general の jvp ルールの性質（issue #81）。
;;;;
;;;; 1 プリミティブだけの graph（make-eqn）を f64 で作り、nb::jvp-graph した
;;;; 結果を中心差分（central-difference-jvp）と比べる。reshape / broadcast-in-dim
;;;; / transpose / reduce-sum / dot-general は主値の変化に対して滑らかなので
;;;; ランダムな配列で比べられる。reduce-max は最大値が重複すると微分できない
;;;; ので、ランダム配列（連続分布なので重複しない）で中心差分と比べ、重複する
;;;; 場合は固定の例で「最大値を取る要素の接線の平均」になること（JAX と同じ）を
;;;; 確かめる。jvp-test.lisp のヘルパー（%jvp-arrays など）を使うので、その後に
;;;; ロードする。dot-general の生成器は tests/primitives/dot-test.lisp のもの。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %prim-graph (prim-name avals &rest params)
  "AVALS を入力とし、PRIM-NAME を PARAMS で1回適用する graph。"
  (let* ((vars (mapcar #'nb::make-var avals))
         (eqn (apply #'nb::make-eqn prim-name vars params)))
    (nb::make-graph vars (list eqn) (nb:eqn-outvars eqn) '())))

(defun %seed-shape (seed rank)
  "SEED から rank RANK・各軸のサイズが 2..4 の shape を作る。"
  (loop for k below rank collect (+ 2 (mod (floor seed (expt 3 k)) 3))))

(defun %seed-subset (seed rank)
  "0..RANK-1 の空でない部分集合（昇順）を SEED から選ぶ。"
  (let ((bits (1+ (mod seed (1- (expt 2 rank))))))
    (loop for i below rank when (logbitp i bits) collect i)))

(defun %shuffle-by-seed (list seed)
  (let ((v (coerce list 'vector)) (state seed))
    (loop for i from (1- (length v)) downto 1
          do (setf state (mod (+ (* state 1103515245) 12345) 2147483648))
             (rotatef (aref v i) (aref v (mod (floor state 16) (1+ i)))))
    (coerce v 'list)))

(defun %broadcast-case-graph (seed)
  "SEED から broadcast-in-dim の graph を作る: オペランドは rank 0〜2 でサイズ1の
次元を含みうる。出力は rank が 1〜2 増え、dims は出力の軸のランダムな部分集合
（昇順）。サイズ1の次元は任意のサイズに広がる。"
  (let* ((op-rank (mod seed 3))
         (out-rank (+ op-rank 1 (mod (floor seed 3) 2)))
         (dims (sort (subseq (%shuffle-by-seed (loop for i below out-rank collect i) seed) 0 op-rank) #'<))
         (op-shape (loop for k below op-rank
                         collect (if (logbitp k (floor seed 7)) 1 (+ 2 (mod (floor seed (+ 11 k)) 3)))))
         (out-shape (loop for i below out-rank
                          collect (let ((k (position i dims)))
                                    (if (and k (> (nth k op-shape) 1))
                                        (nth k op-shape)
                                        (+ 2 (mod (floor seed (+ 5 i)) 3)))))))
    (%prim-graph :broadcast-in-dim (list (nb:make-aval op-shape :f64)) :shape out-shape :dims dims)))

(defun %shape-case-graph (seed)
  "SEED から、(種類 . graph) を選ぶ（f64、dot-general 以外）。"
  (let* ((kind (mod seed 5))
         (rest (floor seed 5))
         (shape (%seed-shape rest 3))
         (aval (nb:make-aval shape :f64))
         (total (reduce #'* shape)))
    (ecase kind
      (0 (%prim-graph :reshape (list aval)
                      :shape (if (evenp rest) (list total) (list (first shape) (/ total (first shape))))))
      (1 (%broadcast-case-graph rest))
      (2 (%prim-graph :transpose (list aval) :perm (%shuffle-by-seed '(0 1 2) rest)))
      ((3 4) (let* ((rank (+ 2 (mod rest 3)))
                    (aval (nb:make-aval (%seed-shape (floor rest 3) rank) :f64)))
               (%prim-graph (if (= kind 3) :reduce-sum :reduce-max) (list aval)
                            :axes (%seed-subset rest rank)))))))

(defun %dot-case-graph (case)
  (destructuring-bind (lhs-shape rhs-shape lc rc lb rb) case
    (%prim-graph :dot-general
                 (list (nb:make-aval lhs-shape :f64) (nb:make-aval rhs-shape :f64))
                 :lhs-contracting lc :rhs-contracting rc :lhs-batch lb :rhs-batch rb)))

(defun %jvp-matches-central-difference-p (graph &optional (seed 0))
  (let* ((primals (%jvp-arrays graph :seed seed))
         (tangents (%jvp-arrays graph :seed seed :tangent t))
         (n-out (length (nb:graph-outvars graph)))
         (jvp (nb::jvp-graph graph)))
    (and (%jvp-round-trips-p jvp)
         (%results-close-p (subseq (%jvp-eval jvp (append primals tangents)) n-out)
                           (central-difference-jvp graph primals tangents)
                           :rtol *autodiff-rtol* :atol *autodiff-atol*))))

(defun %jvp-tangent-is-linear-p (graph)
  (let* ((jvp (nb::jvp-graph graph))
         (n-out (length (nb:graph-outvars graph)))
         (primals (%jvp-arrays graph))
         (v (%jvp-arrays graph :tangent t :seed 0))
         (w (%jvp-arrays graph :tangent t :seed 50)))
    (flet ((tangent-of (tangents) (subseq (%jvp-eval jvp (append primals tangents)) n-out))
           (scaled (arrays) (mapcar (lambda (a) (%scale-array a 3)) arrays)))
      (and (%results-close-p (tangent-of (scaled v)) (scaled (tangent-of v)) :rtol 1d-9 :atol 1d-9)
           (%results-close-p (tangent-of (mapcar #'%sum-array v w))
                             (mapcar #'%sum-array (tangent-of v) (tangent-of w))
                             :rtol 1d-9 :atol 1d-9)))))

(test jvp-shape/matches-central-difference
  "reshape / broadcast-in-dim / transpose / reduce-sum / reduce-max の jvp は
f64 の中心差分と一致する（ランダムな shape・perm・axes）。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed) (%jvp-matches-central-difference-p (%shape-case-graph seed) seed))
                :regression-id jvp-shape/matches-central-difference
                :regression-file (regression-path "jvp-shape-central-difference"))))

(test jvp-shape/tangent-is-linear
  "reshape / broadcast-in-dim / transpose / reduce-sum / reduce-max の jvp は
接線について線形（スケール倍と加法）。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed) (%jvp-tangent-is-linear-p (%shape-case-graph seed)))
                :regression-id jvp-shape/tangent-is-linear
                :regression-file (regression-path "jvp-shape-linear"))))

(test jvp-shape/dot-general-matches-central-difference
  "dot-general の jvp は、ランダムな縮約・バッチ次元の組で f64 の中心差分と一致する。"
  (let ((*num-trials* 30))
    (is (check-it (%dot-generator)
                  (lambda (case) (%jvp-matches-central-difference-p (%dot-case-graph case)))
                  :regression-id jvp-shape/dot-general-matches-central-difference
                  :regression-file (regression-path "jvp-shape-dot-central-difference")))))

(test jvp-shape/dot-general-tangent-is-linear
  "dot-general の jvp は接線について線形。"
  (let ((*num-trials* 30))
    (is (check-it (%dot-generator)
                  (lambda (case) (%jvp-tangent-is-linear-p (%dot-case-graph case)))
                  :regression-id jvp-shape/dot-general-tangent-is-linear
                  :regression-file (regression-path "jvp-shape-dot-linear")))))

(test jvp-shape/dot-general-one-sided-nonzero-drops-the-other-term
  "dot-general で片側の接線だけ非ゼロにした jvp は、もう片側の接線に 0 を入れた
完全な jvp と一致する。"
  (is (check-it (%dot-generator)
                (lambda (case)
                  (let* ((graph (%dot-case-graph case))
                         (full (nb::jvp-graph graph))
                         (primals (%jvp-arrays graph))
                         (tangents (%jvp-arrays graph :tangent t)))
                    (every (lambda (nonzero)
                             (let* ((partial (nb::jvp-graph graph :nonzero nonzero))
                                    (kept (loop for tg in tangents for f in nonzero when f collect tg))
                                    (zeroed (loop for tg in tangents for f in nonzero
                                                  collect (if f tg (%scale-array tg 0)))))
                               (%results-close-p (%jvp-eval partial (append primals kept))
                                                 (%jvp-eval full (append primals zeroed)))))
                           '((t nil) (nil t)))))
                :regression-id jvp-shape/dot-general-one-sided-nonzero
                :regression-file (regression-path "jvp-shape-dot-one-sided"))))

(defun %f64-vector (&rest values)
  (make-array (length values) :element-type 'double-float :initial-contents (mapcar (lambda (x) (coerce x 'double-float)) values)))

(defun %f64-array (dims &rest values)
  (let ((array (make-array dims :element-type 'double-float)))
    (loop for v in values for i from 0 do (setf (row-major-aref array i) (coerce v 'double-float)))
    array))

(test jvp-shape/reduce-max-ties-average-the-tangents
  "最大値が重複するとき、reduce-max の jvp は最大値を取る要素の接線の平均になる
（JAX と同じ）。固定例: [1 3 3 2] で接線 [10 20 40 5] → 30。軸を指定した
2 次元の例: 行ごとの最大 [[2 2 1] [0 5 5]]、接線は行ごとに最大値を取る要素の平均 [2 3]。"
  (let* ((graph (%prim-graph :reduce-max (list (nb:make-aval '(4) :f64)) :axes '(0)))
         (result (%jvp-eval (nb::jvp-graph graph)
                            (list (%f64-vector 1 3 3 2) (%f64-vector 10 20 40 5)))))
    (is (= 3d0 (aref (first result))))
    (is (= 30d0 (aref (second result)))))
  (let* ((graph (%prim-graph :reduce-max (list (nb:make-aval '(2 3) :f64)) :axes '(1)))
         (result (%jvp-eval (nb::jvp-graph graph)
                            (list (%f64-array '(2 3) 2 2 1  0 5 5)
                                  (%f64-array '(2 3) 1 3 7  9 2 4)))))
    (is (equalp (%f64-vector 2 5) (first result)))
    (is (equalp (%f64-vector 2 3) (second result)))))

(test jvp-shape/reduce-max-ties-across-non-reduced-axes
  "同値が縮約しない軸をまたいでも、平均は縮約する軸の中だけで取る。
[[3 3] [3 1]]、axes (1): 行 0 は [3 3] の平均、行 1 は 3 だけ。"
  (let* ((graph (%prim-graph :reduce-max (list (nb:make-aval '(2 2) :f64)) :axes '(1)))
         (result (%jvp-eval (nb::jvp-graph graph)
                            (list (%f64-array '(2 2) 3 3 3 1)
                                  (%f64-array '(2 2) 2 4 10 100)))))
    (is (equalp (%f64-vector 3 3) (first result)))
    (is (equalp (%f64-vector 3 10) (second result)))))

(test jvp-shape/reduce-max-jvp-graph-is-well-formed-for-half-dtypes
  "bf16 / f16 の reduce-max も jvp-graph が通り、check-graph と往復を満たす（評価はしない）。"
  (dolist (dtype '(:bf16 :f16))
    (let ((jvp (nb::jvp-graph (%prim-graph :reduce-max (list (nb:make-aval '(2 3) dtype)) :axes '(1)))))
      (is (%jvp-round-trips-p jvp))
      (is (equalp (list (nb:make-aval '(2) dtype) (nb:make-aval '(2) dtype))
                  (mapcar #'nb:var-aval (nb:graph-outvars jvp)))))))

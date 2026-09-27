;;;; dot-general の性質（issue #31 p5）。
;;;;
;;;; %SHUFFLED / %DISTINCT-DIMS / %AD-HOC-GENERATOR / %ABSTRACT-EVAL-OF /
;;;; %EAGER-OF / %EMIT-OF / %FIXTURE-OP-LINES / %NORMALIZE-SSA-NAMES は
;;;; tests/primitives/shape-test.lisp（同じ nabla.tests パッケージ、p4）で
;;;; 定義済みのものをそのまま再利用する（チェーンBの中、同じテスト
;;;; システムなので DAMP の重複を作らない）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; --- ケース生成: batch / contracting / lhs 専用 free / rhs 専用 free の
;;; 4グループに分けて shape と次元指定を組み立てる ---

(defun %random-dot-case ()
  "(LHS-SHAPE RHS-SHAPE LHS-CONTRACTING RHS-CONTRACTING LHS-BATCH RHS-BATCH)
を返す。batch・contracting・lhs 専用 free・rhs 専用 free の4グループの
次元数の合計は0〜4、サイズはすべて相異なる（%DISTINCT-DIMS と同じ考え方
——次元サイズが同じだと batch/free の入れ替えバグを見逃す。契約 §4 の
mutation の項）。各グループの次元は、lhs・rhs それぞれの shape の中の
ランダムな位置に置く（対応するペアどうしの順序は保つ）。"
  (let* ((total (random 5))
         (group-of (loop repeat total collect (random 4)))
         (n-batch (count 0 group-of))
         (n-contract (count 1 group-of))
         (n-lhs-free (count 2 group-of))
         (n-rhs-free (count 3 group-of))
         (sizes (%distinct-dims total :max 8))
         (batch-sizes (subseq sizes 0 n-batch))
         (contract-sizes (subseq sizes n-batch (+ n-batch n-contract)))
         (lhs-free-sizes (subseq sizes (+ n-batch n-contract) (+ n-batch n-contract n-lhs-free)))
         (rhs-free-sizes (subseq sizes (+ n-batch n-contract n-lhs-free) total))
         (lhs-rank (+ n-batch n-contract n-lhs-free))
         (rhs-rank (+ n-batch n-contract n-rhs-free))
         (lhs-positions (%shuffled (loop for i below lhs-rank collect i)))
         (rhs-positions (%shuffled (loop for i below rhs-rank collect i)))
         (lhs-batch (subseq lhs-positions 0 n-batch))
         (lhs-contracting (subseq lhs-positions n-batch (+ n-batch n-contract)))
         (lhs-free (subseq lhs-positions (+ n-batch n-contract) lhs-rank))
         (rhs-batch (subseq rhs-positions 0 n-batch))
         (rhs-contracting (subseq rhs-positions n-batch (+ n-batch n-contract)))
         (rhs-free (subseq rhs-positions (+ n-batch n-contract) rhs-rank))
         (lhs-shape (make-list lhs-rank))
         (rhs-shape (make-list rhs-rank)))
    (loop for d in lhs-batch for s in batch-sizes do (setf (nth d lhs-shape) s))
    (loop for d in rhs-batch for s in batch-sizes do (setf (nth d rhs-shape) s))
    (loop for d in lhs-contracting for s in contract-sizes do (setf (nth d lhs-shape) s))
    (loop for d in rhs-contracting for s in contract-sizes do (setf (nth d rhs-shape) s))
    (loop for d in lhs-free for s in lhs-free-sizes do (setf (nth d lhs-shape) s))
    (loop for d in rhs-free for s in rhs-free-sizes do (setf (nth d rhs-shape) s))
    (list lhs-shape rhs-shape lhs-contracting rhs-contracting lhs-batch rhs-batch)))

(defun %dot-generator () (%ad-hoc-generator #'%random-dot-case))

;;; ============================ 性質 ============================

(test dot-general/aval-matches-eager
  "abstract-eval が返す aval は、eager を実際に実行した結果の aval と、
すべての float dtype で一致する。"
  (is (check-it (%dot-generator)
                (lambda (case)
                  (destructuring-bind (lhs-shape rhs-shape lhs-contracting rhs-contracting lhs-batch rhs-batch) case
                    (every (lambda (dtype)
                             (let* ((lhs (make-random-array (make-array-spec lhs-shape dtype)))
                                    (rhs (make-random-array (make-array-spec rhs-shape dtype) :seed 1))
                                    (lhs-aval (nb:array-aval lhs dtype))
                                    (rhs-aval (nb:array-aval rhs dtype))
                                    (expected-aval (%abstract-eval-of :dot-general (list lhs-aval rhs-aval)
                                                                       :lhs-contracting lhs-contracting
                                                                       :rhs-contracting rhs-contracting
                                                                       :lhs-batch lhs-batch :rhs-batch rhs-batch))
                                    (result (%eager-of :dot-general (list lhs rhs) (list lhs-aval rhs-aval)
                                                        :lhs-contracting lhs-contracting :rhs-contracting rhs-contracting
                                                        :lhs-batch lhs-batch :rhs-batch rhs-batch)))
                               (equalp expected-aval (nb:array-aval result (nb:aval-dtype expected-aval)))))
                           *dtypes*)))
                :regression-id dot-general/aval-matches-eager
                :regression-file (regression-path "dot-general-aval"))))

(test dot-general/eager-matches-reference
  "eager 実装を decode-array で double-float に戻した結果は、
reference-dot-general の期待値と dtype ごとの許容誤差で一致する。"
  (is (check-it (%dot-generator)
                (lambda (case)
                  (destructuring-bind (lhs-shape rhs-shape lhs-contracting rhs-contracting lhs-batch rhs-batch) case
                    (every (lambda (dtype)
                             (let* ((lhs (make-random-array (make-array-spec lhs-shape dtype)))
                                    (rhs (make-random-array (make-array-spec rhs-shape dtype) :seed 1))
                                    (lhs-aval (nb:array-aval lhs dtype))
                                    (rhs-aval (nb:array-aval rhs dtype))
                                    (result (%eager-of :dot-general (list lhs rhs) (list lhs-aval rhs-aval)
                                                        :lhs-contracting lhs-contracting :rhs-contracting rhs-contracting
                                                        :lhs-batch lhs-batch :rhs-batch rhs-batch))
                                    (expected (reference-dot-general (decode-array lhs dtype) (decode-array rhs dtype)
                                                                      :lhs-contracting lhs-contracting :rhs-contracting rhs-contracting
                                                                      :lhs-batch lhs-batch :rhs-batch rhs-batch)))
                               (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                 (allclose (decode-array result dtype) expected :rtol rtol :atol atol))))
                           *dtypes*)))
                :regression-id dot-general/eager-matches-reference
                :regression-file (regression-path "dot-general-eager"))))

(test dot-general/no-batch-matches-reference-matmul
  "batch なし・rank 2 の dot-general（contracting (1)/(0)）は
reference-matmul と f64 の許容誤差で一致する。"
  (let* ((lhs (make-random-array (make-array-spec '(2 3) :f64)))
         (rhs (make-random-array (make-array-spec '(3 4) :f64) :seed 2))
         (result (%eager-of :dot-general (list lhs rhs)
                             (list (nb:array-aval lhs :f64) (nb:array-aval rhs :f64))
                             :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
         (expected (reference-matmul lhs rhs)))
    (multiple-value-bind (rtol atol) (dtype-tolerance :f64)
      (is (allclose result expected :rtol rtol :atol atol)))))

(test dot-general/empty-contracting-is-outer-product
  "contracting が空リストの dot-general は外積になる: 結果の各要素は
対応する lhs・rhs の要素の積そのもの。"
  (let* ((lhs (make-random-array (make-array-spec '(2) :f32)))
         (rhs (make-random-array (make-array-spec '(3) :f32) :seed 3))
         (result (%eager-of :dot-general (list lhs rhs)
                             (list (nb:array-aval lhs :f32) (nb:array-aval rhs :f32))
                             :lhs-contracting '() :rhs-contracting '() :lhs-batch '() :rhs-batch '())))
    (is (equal '(2 3) (array-dimensions result)))
    (dotimes (i 2)
      (dotimes (j 3)
        (is (= (aref result i j) (* (aref lhs i) (aref rhs j))))))))

(test dot-general/rank0-result-fully-contracted
  "両方の operand を完全に contract すると rank 0（内積）になる。"
  (let* ((lhs (make-random-array (make-array-spec '(4) :f32)))
         (rhs (make-random-array (make-array-spec '(4) :f32) :seed 4))
         (result (%eager-of :dot-general (list lhs rhs)
                             (list (nb:array-aval lhs :f32) (nb:array-aval rhs :f32))
                             :lhs-contracting '(0) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
         (expected (loop for i below 4 sum (* (aref lhs i) (aref rhs i)))))
    (is (equal '() (array-dimensions result)))
    (multiple-value-bind (rtol atol) (dtype-tolerance :f32)
      (is (approx= (aref result) expected :rtol rtol :atol atol)))))

(test dot-general/batched-matches-per-batch-unbatched-dot
  "batch 付きの dot-general は、バッチ添字ごとに対応する slice を
batch なしの dot-general にかけてループした結果（独立オラクル）と
f64 の許容誤差で一致する。"
  (let* ((batch 2) (m 3) (k 4) (n 2)
         (lhs (make-random-array (make-array-spec (list batch m k) :f64)))
         (rhs (make-random-array (make-array-spec (list batch k n) :f64) :seed 5))
         (result (%eager-of :dot-general (list lhs rhs)
                             (list (nb:array-aval lhs :f64) (nb:array-aval rhs :f64))
                             :lhs-contracting '(2) :rhs-contracting '(1) :lhs-batch '(0) :rhs-batch '(0))))
    (is (equal (list batch m n) (array-dimensions result)))
    (dotimes (b batch)
      (let* ((lhs-slice (make-array (list m k) :element-type 'double-float))
             (rhs-slice (make-array (list k n) :element-type 'double-float)))
        (dotimes (i m) (dotimes (p k) (setf (aref lhs-slice i p) (aref lhs b i p))))
        (dotimes (p k) (dotimes (j n) (setf (aref rhs-slice p j) (aref rhs b p j))))
        (let ((expected-slice (reference-matmul lhs-slice rhs-slice)))
          (dotimes (i m)
            (dotimes (j n)
              (multiple-value-bind (rtol atol) (dtype-tolerance :f64)
                (is (approx= (aref result b i j) (aref expected-slice i j) :rtol rtol :atol atol))))))))))

(test dot-general/linearity-in-f64
  "dot(a+b, c) = dot(a,c) + dot(b,c)（f64、rank 2、batch なし）。"
  (let* ((a (make-random-array (make-array-spec '(2 3) :f64)))
         (b (make-random-array (make-array-spec '(2 3) :f64) :seed 6))
         (c (make-random-array (make-array-spec '(3 4) :f64) :seed 7))
         (a-plus-b (make-array '(2 3) :element-type 'double-float)))
    (dotimes (i 2) (dotimes (j 3) (setf (aref a-plus-b i j) (+ (aref a i j) (aref b i j)))))
    (let* ((lhs-of-sum (%eager-of :dot-general (list a-plus-b c)
                                   (list (nb:array-aval a-plus-b :f64) (nb:array-aval c :f64))
                                   :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
           (dot-a (%eager-of :dot-general (list a c) (list (nb:array-aval a :f64) (nb:array-aval c :f64))
                              :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
           (dot-b (%eager-of :dot-general (list b c) (list (nb:array-aval b :f64) (nb:array-aval c :f64))
                              :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
           (sum-of-dots (make-array '(2 4) :element-type 'double-float)))
      (dotimes (i 2) (dotimes (j 4) (setf (aref sum-of-dots i j) (+ (aref dot-a i j) (aref dot-b i j)))))
      (multiple-value-bind (rtol atol) (dtype-tolerance :f64)
        (is (allclose lhs-of-sum sum-of-dots :rtol rtol :atol atol))))))

(test dot-general/transposition-symmetry
  "dot(a,b) を contracting (1)/(0) で計算した結果は、dot(b,a) を同じ
contracting (1)/(0) で計算した結果を transpose した行列に等しい
（transpose の eager 実装をオラクルに使う）。"
  (let* ((a (make-random-array (make-array-spec '(2 3) :f32)))
         (b (make-random-array (make-array-spec '(3 4) :f32) :seed 8))
         (ab (%eager-of :dot-general (list a b) (list (nb:array-aval a :f32) (nb:array-aval b :f32))
                         :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
         (ba (%eager-of :dot-general (list b a) (list (nb:array-aval b :f32) (nb:array-aval a :f32))
                         :lhs-contracting '(0) :rhs-contracting '(1) :lhs-batch '() :rhs-batch '()))
         (ba-transposed (%eager-of :transpose (list ba) (list (nb:array-aval ba :f32)) :perm '(1 0))))
    (multiple-value-bind (rtol atol) (dtype-tolerance :f32)
      (is (allclose ab ba-transposed :rtol rtol :atol atol)))))

;;; ======================= 不正な入力・境界値 =======================

(test dot-general/wrong-arity-signals-primitive-error
  "入力が0個・1個・3個の dot-general は PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error
    (%abstract-eval-of :dot-general '() :lhs-contracting '() :rhs-contracting '() :lhs-batch '() :rhs-batch '()))
  (signals nb:primitive-error
    (%abstract-eval-of :dot-general (list (nb:make-aval '(2 3) :f32))
                       :lhs-contracting '() :rhs-contracting '() :lhs-batch '() :rhs-batch '()))
  (signals nb:primitive-error
    (%abstract-eval-of :dot-general
                       (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(3 2) :f32) (nb:make-aval '(2) :f32))
                       :lhs-contracting '() :rhs-contracting '() :lhs-batch '() :rhs-batch '())))

(test dot-general/non-float-dtype-signals-primitive-error
  "float でない dtype（:i1）の入力は PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error
    (%abstract-eval-of :dot-general (list (nb:make-aval '(2 3) :i1) (nb:make-aval '(3 2) :i1))
                       :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '())))

(test dot-general/mismatched-dtypes-signal-primitive-error
  "2つの入力の dtype が違えば PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error
    (%abstract-eval-of :dot-general (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(3 2) :f64))
                       :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '())))

(test dot-general/mismatched-list-lengths-signal-primitive-error
  "lhs-contracting と rhs-contracting、lhs-batch と rhs-batch の長さが
違えば PRIMITIVE-ERROR になる。"
  (let ((in-avals (list (nb:make-aval '(2 3 4) :f32) (nb:make-aval '(2 4 5) :f32))))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general in-avals :lhs-contracting '(2) :rhs-contracting '(1 2) :lhs-batch '(0) :rhs-batch '(0)))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general in-avals :lhs-contracting '(2) :rhs-contracting '(1) :lhs-batch '(0) :rhs-batch '()))))

(test dot-general/out-of-range-index-signals-primitive-error
  "境界: index = rank（範囲外）を含む contracting/batch は PRIMITIVE-ERROR
になる。"
  (let ((in-avals (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(3 2) :f32))))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general in-avals :lhs-contracting '(2) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general in-avals :lhs-contracting '(1) :rhs-contracting '(2) :lhs-batch '() :rhs-batch '()))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general in-avals :lhs-contracting '(-1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))))

(test dot-general/mismatched-contracting-size-signals-primitive-error
  "対応する contracting dim のサイズが違えば PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error
    (%abstract-eval-of :dot-general (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(4 2) :f32))
                       :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '())))

(test dot-general/mismatched-batch-size-signals-primitive-error
  "対応する batch dim のサイズが違えば PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error
    (%abstract-eval-of :dot-general (list (nb:make-aval '(2 3 4) :f32) (nb:make-aval '(3 4 5) :f32))
                       :lhs-contracting '(2) :rhs-contracting '(1) :lhs-batch '(0) :rhs-batch '(0))))

(test dot-general/duplicated-dim-signals-primitive-error
  "同じ次元が contracting と batch の両方、あるいは同じリストの中で
重複していれば PRIMITIVE-ERROR になる。"
  (let ((in-avals (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32))))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general in-avals :lhs-contracting '(0) :rhs-contracting '(0) :lhs-batch '(0) :rhs-batch '(1)))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general in-avals :lhs-contracting '(0 0) :rhs-contracting '(0 1) :lhs-batch '() :rhs-batch '()))))

;;; ============================== emit ==============================

(test dot-general/emit-matches-fixture
  "batch なしの :emit（f32・bf16）は、SSA 名を正規化した後
dot_general.mlir / dot_general_bf16.mlir の op 行と一致する。"
  (is (string= (%normalize-ssa-names
                (%emit-of :dot-general '("%a" "%b")
                          (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(3 2) :f32))
                          "%0" (nb:make-aval '(2 2) :f32)
                          :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
               (%normalize-ssa-names (first (%fixture-op-lines "dot_general")))))
  (is (string= (%normalize-ssa-names
                (%emit-of :dot-general '("%a" "%b")
                          (list (nb:make-aval '(2 3) :bf16) (nb:make-aval '(3 2) :bf16))
                          "%0" (nb:make-aval '(2 2) :bf16)
                          :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
               (%normalize-ssa-names (first (%fixture-op-lines "dot_general_bf16"))))))

(test dot-general/emit-with-batch-spells-batching-dims
  "batch ありの :emit は batching_dims と contracting_dims の両方を、
契約どおりの綴りで1行に出す。"
  (is (string= (%emit-of :dot-general '("%a" "%b")
                          (list (nb:make-aval '(2 3 4) :f32) (nb:make-aval '(2 4 5) :f32))
                          "%0" (nb:make-aval '(2 3 5) :f32)
                          :lhs-contracting '(2) :rhs-contracting '(1) :lhs-batch '(0) :rhs-batch '(0))
               "%0 = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>")))

(test dot-general/emit-empty-contracting-spells-empty-brackets
  "contracting が空リストの :emit は `contracting_dims = [] x []` と
batching_dims 無しで出す。"
  (is (string= (%emit-of :dot-general '("%a" "%b")
                          (list (nb:make-aval '(2) :f32) (nb:make-aval '(3) :f32))
                          "%0" (nb:make-aval '(2 3) :f32)
                          :lhs-contracting '() :rhs-contracting '() :lhs-batch '() :rhs-batch '())
               "%0 = stablehlo.dot_general %a, %b, contracting_dims = [] x [] : (tensor<2xf32>, tensor<3xf32>) -> tensor<2x3xf32>")))

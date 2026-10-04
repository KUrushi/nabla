;;;; dot-general の性質（issue #31 p5）。
;;;;
;;;; find-primitive / primitive-abstract-eval / primitive-emit /
;;;; primitive-eager は内部シンボル（nb::）で呼ぶ（他の primitive テストと
;;;; 同じ流儀。tests/primitives/shape-test.lisp 参照）。%FIXTURE-OP-LINES /
;;;; %NORMALIZE-SSA-NAMES / %EAGER-OF などは shape-test.lisp と同じ内容を
;;;; このファイルにも持つ（契約 §4 テスト点4: 各ユニットが自分の
;;;; コピーを定義する、受け入れられた DAMP 重複。wave 3 で dedupe する）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; --- primitive の呼び出しを短く書くためのヘルパー ---

(defun %abstract-eval-of (name in-avals &rest params)
  (apply (nb::primitive-abstract-eval (nb::find-primitive name)) in-avals params))

(defun %eager-of (name arrays in-avals &rest params)
  (apply (nb::primitive-eager (nb::find-primitive name)) arrays in-avals params))

(defun %emit-of (name in-names in-avals out-name out-aval &rest params)
  (apply (nb::primitive-emit (nb::find-primitive name)) in-names in-avals out-name out-aval params))

;;; --- フィクスチャの op 行の取り出し ---

(defun %fixture-text (op-name)
  (let ((path (asdf:system-relative-pathname
               "nabla" (format nil "tests/fixtures/stablehlo/ops/~A.mlir" op-name))))
    (with-open-file (stream path :direction :input)
      (let ((text (make-string (file-length stream))))
        (subseq text 0 (read-sequence text stream))))))

(defun %fixture-op-lines (op-name)
  "tests/fixtures/stablehlo/ops/OP-NAME.mlir の func.func 行と func.return 行の
間にある行を、前後の空白を落として返す。"
  (let* ((lines (uiop:split-string (%fixture-text op-name) :separator '(#\Newline)))
         (start (position-if (lambda (l) (search "func.func" l)) lines))
         (end (position-if (lambda (l) (search "func.return" l)) lines)))
    (mapcar (lambda (l) (string-trim '(#\Space #\Tab) l)) (subseq lines (1+ start) end))))

(defun %normalize-ssa-names (text)
  "TEXT 中の %[A-Za-z0-9_]+ という形のトークンを、すべて \"%\" 1文字に
置き換える（golden emit テストで SSA 名の違いを無視するため）。"
  (with-output-to-string (out)
    (let ((i 0) (n (length text)))
      (loop while (< i n)
            do (if (char= (char text i) #\%)
                   (progn
                     (write-char #\% out)
                     (incf i)
                     (loop while (and (< i n)
                                      (let ((c (char text i)))
                                        (or (alphanumericp c) (char= c #\_))))
                           do (incf i)))
                   (progn (write-char (char text i) out) (incf i)))))))

;;; --- ランダムな (lhs-shape rhs-shape lhs-contracting rhs-contracting
;;; lhs-batch rhs-batch) の組を作る generator ---

(defun %dot-shuffled (list)
  "LIST の要素をランダムに並べ替えたリストを返す（Fisher-Yates）。"
  (let ((v (coerce list 'vector)))
    (loop for i from (1- (length v)) downto 1
          do (rotatef (aref v i) (aref v (random (1+ i)))))
    (coerce v 'list)))

(defun %dot-distinct-sizes (n &key (max 8))
  "1..MAX から相異なる N 個のサイズを選ぶ。mutation testing でストライド・
添字計算の +/- や次元の入れ替えを殺すには、バッチ・縮約・自由次元の
サイズがすべて異なっていないと同じ結果になってしまうケースが多い
（契約 §4 の mutation の項）。MAX を小さく保つのは、配列の総要素数が
グループの rank ぶんの相異なる整数の積になり、大きくすると PBT が
何百回も巨大な配列を作ってしまうため（rank は各グループ 0〜2 に抑え、
片側あたり最大6次元・積は高々 3*4*5*6*7*8=20160 程度に収める）。"
  (subseq (%dot-shuffled (loop for d from 1 to max collect d)) 0 n))

(defun %random-dot-case ()
  "(lhs-shape rhs-shape lhs-contracting rhs-contracting lhs-batch rhs-batch)
を返す。バッチ・縮約・lhs 自由・rhs 自由の各グループの rank は 0〜2、
サイズはグループをまたいですべて相異なる。"
  (let* ((batch-rank (random 3))
         (contract-rank (random 3))
         (lhs-free-rank (random 3))
         (rhs-free-rank (random 3))
         (total (+ batch-rank contract-rank lhs-free-rank rhs-free-rank))
         (sizes (%dot-distinct-sizes total))
         (batch-sizes (subseq sizes 0 batch-rank))
         (contract-sizes (subseq sizes batch-rank (+ batch-rank contract-rank)))
         (lhs-free-sizes (subseq sizes (+ batch-rank contract-rank)
                                 (+ batch-rank contract-rank lhs-free-rank)))
         (rhs-free-sizes (subseq sizes (+ batch-rank contract-rank lhs-free-rank) total))
         (lhs-rank (+ batch-rank contract-rank lhs-free-rank))
         (rhs-rank (+ batch-rank contract-rank rhs-free-rank))
         (lhs-positions (%dot-shuffled (loop for i below lhs-rank collect i)))
         (rhs-positions (%dot-shuffled (loop for i below rhs-rank collect i)))
         (lhs-batch (subseq lhs-positions 0 batch-rank))
         (lhs-contracting (subseq lhs-positions batch-rank (+ batch-rank contract-rank)))
         (lhs-free-positions (sort (copy-list (subseq lhs-positions (+ batch-rank contract-rank))) #'<))
         (rhs-batch (subseq rhs-positions 0 batch-rank))
         (rhs-contracting (subseq rhs-positions batch-rank (+ batch-rank contract-rank)))
         (rhs-free-positions (sort (copy-list (subseq rhs-positions (+ batch-rank contract-rank))) #'<))
         (lhs-shape (make-list lhs-rank))
         (rhs-shape (make-list rhs-rank)))
    (loop for pos in lhs-batch for s in batch-sizes do (setf (nth pos lhs-shape) s))
    (loop for pos in lhs-contracting for s in contract-sizes do (setf (nth pos lhs-shape) s))
    (loop for pos in lhs-free-positions for s in lhs-free-sizes do (setf (nth pos lhs-shape) s))
    (loop for pos in rhs-batch for s in batch-sizes do (setf (nth pos rhs-shape) s))
    (loop for pos in rhs-contracting for s in contract-sizes do (setf (nth pos rhs-shape) s))
    (loop for pos in rhs-free-positions for s in rhs-free-sizes do (setf (nth pos rhs-shape) s))
    (list lhs-shape rhs-shape lhs-contracting rhs-contracting lhs-batch rhs-batch)))

(defun %dot-generator () (%ad-hoc-generator #'%random-dot-case))

;;; ============================ aval / eager ============================

(test dot-general/aval-matches-eager
  "dot-general の abstract-eval が返す aval は、eager を実際に実行した
結果の aval と、すべての float dtype で一致する。"
  (is (check-it (%dot-generator)
                (lambda (case)
                  (destructuring-bind (lhs-shape rhs-shape lc rc lb rb) case
                    (every (lambda (dtype)
                             (let* ((lhs (make-random-array (make-array-spec lhs-shape dtype)))
                                    (rhs (make-random-array (make-array-spec rhs-shape dtype) :seed 1))
                                    (lhs-aval (nb:array-aval lhs dtype))
                                    (rhs-aval (nb:array-aval rhs dtype))
                                    (expected-aval (%abstract-eval-of :dot-general (list lhs-aval rhs-aval)
                                                                       :lhs-contracting lc :rhs-contracting rc
                                                                       :lhs-batch lb :rhs-batch rb))
                                    (result (%eager-of :dot-general (list lhs rhs) (list lhs-aval rhs-aval)
                                                        :lhs-contracting lc :rhs-contracting rc
                                                        :lhs-batch lb :rhs-batch rb)))
                               (equalp expected-aval (nb:array-aval result (nb:aval-dtype expected-aval)))))
                           *dtypes*)))
                :regression-id dot-general/aval-matches-eager
                :regression-file (regression-path "dot-general-aval"))))

(test dot-general/eager-matches-reference
  "dot-general の eager 実装を decode-array で double-float に戻した結果は、
reference-dot-general の期待値と dtype ごとの許容誤差で一致する。"
  (is (check-it (%dot-generator)
                (lambda (case)
                  (destructuring-bind (lhs-shape rhs-shape lc rc lb rb) case
                    (every (lambda (dtype)
                             (let* ((lhs (make-random-array (make-array-spec lhs-shape dtype)))
                                    (rhs (make-random-array (make-array-spec rhs-shape dtype) :seed 1))
                                    (lhs-aval (nb:array-aval lhs dtype))
                                    (rhs-aval (nb:array-aval rhs dtype))
                                    (result (%eager-of :dot-general (list lhs rhs) (list lhs-aval rhs-aval)
                                                        :lhs-contracting lc :rhs-contracting rc
                                                        :lhs-batch lb :rhs-batch rb))
                                    (expected (reference-dot-general (decode-array lhs dtype) (decode-array rhs dtype)
                                                                      lc rc lb rb)))
                               (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                 (allclose (decode-array result dtype) expected :rtol rtol :atol atol))))
                           *dtypes*)))
                :regression-id dot-general/eager-matches-reference
                :regression-file (regression-path "dot-general-eager"))))

;;; ===================== オラクルとの比較（PBT） =====================
;;;
;;; スキルの「例ベースのテストは JAX フィクスチャと PBT の回帰テストに
;;; 限る」という決まりに従い、以下はすべて check-it でランダムな形状を
;;; 生成してから比べる（固定した1例だけでは規約に反する）。

(test dot-general/rank2-no-batch-matches-reference-matmul
  "batch 無しの rank 2 の dot-general は reference-matmul と一致する。"
  (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 6)
                                  (uniform-integer :lo 1 :hi 6)
                                  (uniform-integer :lo 1 :hi 6)))
                (lambda (dims)
                  (destructuring-bind (m k n) dims
                    (let* ((lhs (make-random-array (make-array-spec (list m k) :f64)))
                           (rhs (make-random-array (make-array-spec (list k n) :f64) :seed 2))
                           (result (%eager-of :dot-general (list lhs rhs)
                                               (list (nb:array-aval lhs :f64) (nb:array-aval rhs :f64))
                                               :lhs-contracting '(1) :rhs-contracting '(0)
                                               :lhs-batch '() :rhs-batch '())))
                      (allclose (decode-array result :f64) (reference-matmul lhs rhs) :dtype :f64))))
                :regression-id dot-general/rank2-no-batch-matches-reference-matmul
                :regression-file (regression-path "dot-general-matches-reference-matmul"))))

(test dot-general/empty-contracting-is-outer-product
  "縮約次元が空なら外積になる: 出力の各要素は対応する lhs・rhs の要素の
積に一致する。"
  (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 4) (uniform-integer :lo 1 :hi 4)
                                  (uniform-integer :lo 1 :hi 4) (uniform-integer :lo 1 :hi 4)))
                (lambda (dims)
                  (destructuring-bind (d0 d1 d2 d3) dims
                    (let* ((lhs (make-random-array (make-array-spec (list d0 d1) :f32)))
                           (rhs (make-random-array (make-array-spec (list d2 d3) :f32) :seed 3))
                           (result (%eager-of :dot-general (list lhs rhs)
                                               (list (nb:array-aval lhs :f32) (nb:array-aval rhs :f32))
                                               :lhs-contracting '() :rhs-contracting '()
                                               :lhs-batch '() :rhs-batch '())))
                      (and (equalp (list d0 d1 d2 d3) (array-dimensions result))
                           (loop for i0 below d0 always
                                 (loop for i1 below d1 always
                                       (loop for j0 below d2 always
                                             (loop for j1 below d3 always
                                                   (approx= (aref result i0 i1 j0 j1)
                                                            (* (aref lhs i0 i1) (aref rhs j0 j1))
                                                            :dtype :f32)))))))))
                :regression-id dot-general/empty-contracting-is-outer-product
                :regression-file (regression-path "dot-general-outer-product"))))

(test dot-general/rank0-fully-contracted
  "両オペランドとも完全に縮約されると rank 0 の内積になる。"
  (is (check-it (generator (uniform-integer :lo 1 :hi 8))
                (lambda (n)
                  (let* ((lhs (make-random-array (make-array-spec (list n) :f64)))
                         (rhs (make-random-array (make-array-spec (list n) :f64) :seed 4))
                         (result (%eager-of :dot-general (list lhs rhs)
                                             (list (nb:array-aval lhs :f64) (nb:array-aval rhs :f64))
                                             :lhs-contracting '(0) :rhs-contracting '(0)
                                             :lhs-batch '() :rhs-batch '()))
                         (expected (let ((sum 0.0d0)) (dotimes (i n sum) (incf sum (* (aref lhs i) (aref rhs i)))))))
                    (and (equalp '() (array-dimensions result))
                         (approx= (aref result) expected :dtype :f64))))
                :regression-id dot-general/rank0-fully-contracted
                :regression-file (regression-path "dot-general-rank0-fully-contracted"))))

(test dot-general/batched-matches-per-slice-loop
  "バッチ付きの dot-general の結果は、バッチ次元ごとに batch 無しの
dot-general をスライスに適用してループで積み上げた結果と一致する
（バッチ配線と縮約ロジックを独立に確かめる）。"
  (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 4) (uniform-integer :lo 1 :hi 4)
                                  (uniform-integer :lo 1 :hi 4) (uniform-integer :lo 1 :hi 4)))
                (lambda (dims)
                  (destructuring-bind (batch m k n) dims
                    (let* ((lhs (make-random-array (make-array-spec (list batch m k) :f32)))
                           (rhs (make-random-array (make-array-spec (list batch k n) :f32) :seed 5))
                           (lhs-aval (nb:array-aval lhs :f32))
                           (rhs-aval (nb:array-aval rhs :f32))
                           (result (%eager-of :dot-general (list lhs rhs) (list lhs-aval rhs-aval)
                                               :lhs-contracting '(2) :rhs-contracting '(1)
                                               :lhs-batch '(0) :rhs-batch '(0))))
                      (and (equalp (list batch m n) (array-dimensions result))
                           (loop for b below batch always
                                 (let* ((lhs-slice (make-array (list m k) :element-type 'single-float))
                                        (rhs-slice (make-array (list k n) :element-type 'single-float)))
                                   (dotimes (i m) (dotimes (p k) (setf (aref lhs-slice i p) (aref lhs b i p))))
                                   (dotimes (p k) (dotimes (j n) (setf (aref rhs-slice p j) (aref rhs b p j))))
                                   (let ((slice-result (%eager-of :dot-general (list lhs-slice rhs-slice)
                                                                   (list (nb:array-aval lhs-slice :f32)
                                                                         (nb:array-aval rhs-slice :f32))
                                                                   :lhs-contracting '(1) :rhs-contracting '(0)
                                                                   :lhs-batch '() :rhs-batch '())))
                                     (loop for i below m always
                                           (loop for j below n always
                                                 (approx= (aref result b i j) (aref slice-result i j) :dtype :f32))))))))))
                :regression-id dot-general/batched-matches-per-slice-loop
                :regression-file (regression-path "dot-general-batched-matches-per-slice-loop"))))

(test dot-general/linearity-in-f64
  "線形性: dot(a+b, c) = dot(a, c) + dot(b, c)（f64 で計算）。"
  (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 6)
                                  (uniform-integer :lo 1 :hi 6)
                                  (uniform-integer :lo 1 :hi 6)))
                (lambda (dims)
                  (destructuring-bind (m k n) dims
                    (let* ((a (make-random-array (make-array-spec (list m k) :f64)))
                           (b (make-random-array (make-array-spec (list m k) :f64) :seed 6))
                           (c (make-random-array (make-array-spec (list k n) :f64) :seed 7))
                           (sum (make-array (list m k) :element-type 'double-float)))
                      (dotimes (i m) (dotimes (j k) (setf (aref sum i j) (+ (aref a i j) (aref b i j)))))
                      (flet ((dot (x y) (%eager-of :dot-general (list x y)
                                                    (list (nb:array-aval x :f64) (nb:array-aval y :f64))
                                                    :lhs-contracting '(1) :rhs-contracting '(0)
                                                    :lhs-batch '() :rhs-batch '())))
                        (let ((lhs-side (dot sum c))
                              (rhs-side (let ((da (dot a c)) (db (dot b c))
                                              (out (make-array (list m n) :element-type 'double-float)))
                                          (dotimes (i m out)
                                            (dotimes (j n) (setf (aref out i j) (+ (aref da i j) (aref db i j))))))))
                          (allclose lhs-side rhs-side :dtype :f64))))))
                :regression-id dot-general/linearity-in-f64
                :regression-file (regression-path "dot-general-linearity"))))

(test dot-general/transpose-symmetry
  "A(m,k) @ B(k,n) = transpose(B(k,n) @ A(m,k) の縮約を入れ替えたもの)
（p4 の transpose の eager をオラクルにする）。A @ B は (m,n)、
B の縮約を dim0（k）、A の縮約を dim1（k）にした dot(b, a) は (n,m) に
なるので、それを transpose すれば (m,n) に戻り、A @ B と一致するはず。"
  (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 6)
                                  (uniform-integer :lo 1 :hi 6)
                                  (uniform-integer :lo 1 :hi 6)))
                (lambda (dims)
                  (destructuring-bind (m k n) dims
                    (let* ((a (make-random-array (make-array-spec (list m k) :f64)))
                           (b (make-random-array (make-array-spec (list k n) :f64) :seed 8))
                           (ab (%eager-of :dot-general (list a b) (list (nb:array-aval a :f64) (nb:array-aval b :f64))
                                          :lhs-contracting '(1) :rhs-contracting '(0)
                                          :lhs-batch '() :rhs-batch '()))
                           (ba (%eager-of :dot-general (list b a) (list (nb:array-aval b :f64) (nb:array-aval a :f64))
                                          :lhs-contracting '(0) :rhs-contracting '(1)
                                          :lhs-batch '() :rhs-batch '()))
                           (ba-transposed (%eager-of :transpose (list ba) (list (nb:array-aval ba :f64)) :perm '(1 0))))
                      (allclose ab ba-transposed :dtype :f64))))
                :regression-id dot-general/transpose-symmetry
                :regression-file (regression-path "dot-general-transpose-symmetry"))))

;;; ============================ 不正な入力 ============================

(test dot-general/wrong-arity-signals-primitive-error
  "入力が0個・1個・3個の dot-general は PRIMITIVE-ERROR になる。"
  (let ((aval (nb:make-aval '(2 3) :f32)))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general '() :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list aval) :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list aval aval aval)
                         :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))))

(test dot-general/dtype-mismatch-signals-primitive-error
  "dtype が違う、または float でない入力は PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error
    (%abstract-eval-of :dot-general (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(3 2) :f64))
                       :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
  (signals nb:primitive-error
    (%abstract-eval-of :dot-general (list (nb:make-aval '(2 3) :i1) (nb:make-aval '(3 2) :i1))
                       :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '())))

(test dot-general/mismatched-list-lengths-signal-primitive-error
  "lhs-contracting と rhs-contracting、lhs-batch と rhs-batch の長さが
違えば PRIMITIVE-ERROR になる。"
  (let ((lhs (nb:make-aval '(2 3 4) :f32)) (rhs (nb:make-aval '(2 4 5) :f32)))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list lhs rhs) :lhs-contracting '(2) :rhs-contracting '(1 0)
                         :lhs-batch '(0) :rhs-batch '(0)))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list lhs rhs) :lhs-contracting '(2) :rhs-contracting '(1)
                         :lhs-batch '(0) :rhs-batch '()))))

(test dot-general/out-of-range-index-signals-primitive-error
  "index = rank と index = -1（どちらも境界外）は PRIMITIVE-ERROR になる
（-1 は、範囲の下限を (integer -1 ...) に緩める変異体が issue #70 の
mutation testing で生き残っていたため足した）。"
  (let ((lhs (nb:make-aval '(2 3) :f32)) (rhs (nb:make-aval '(3 2) :f32)))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list lhs rhs) :lhs-contracting '(2) :rhs-contracting '(0)
                         :lhs-batch '() :rhs-batch '()))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list lhs rhs) :lhs-contracting '(-1) :rhs-contracting '(0)
                         :lhs-batch '() :rhs-batch '()))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list lhs rhs) :lhs-contracting '(1) :rhs-contracting '(2)
                         :lhs-batch '() :rhs-batch '()))))

(test dot-general/duplicated-dim-signals-primitive-error
  "同じ次元が batch と contracting の両方、または同じリストに重複して
現れると PRIMITIVE-ERROR になる。"
  (let ((lhs (nb:make-aval '(2 3 4) :f32)) (rhs (nb:make-aval '(2 3 4) :f32)))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list lhs rhs) :lhs-contracting '(0) :rhs-contracting '(0)
                         :lhs-batch '(0) :rhs-batch '(1)))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list lhs rhs) :lhs-contracting '(0 0) :rhs-contracting '(0 1)
                         :lhs-batch '() :rhs-batch '()))))

(test dot-general/mismatched-dim-sizes-signal-primitive-error
  "対応する batch / contracting 次元のサイズが違えば PRIMITIVE-ERROR に
なる。"
  (signals nb:primitive-error
    (%abstract-eval-of :dot-general (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(4 2) :f32))
                       :lhs-contracting '(1) :rhs-contracting '(1) :lhs-batch '() :rhs-batch '()))
  (signals nb:primitive-error
    (%abstract-eval-of :dot-general (list (nb:make-aval '(2 3 4) :f32) (nb:make-aval '(5 4 6) :f32))
                       :lhs-contracting '(2) :rhs-contracting '(1) :lhs-batch '(0) :rhs-batch '(0))))

(test dot-general/malformed-param-list-signals-primitive-error
  "lhs-contracting / rhs-contracting / lhs-batch / rhs-batch がドットリスト
（例: '(1 . 2)）や非整数要素を含むリストのときも、生の LENGTH の
TYPE-ERROR ではなく PRIMITIVE-ERROR になる（契約 §0）。"
  (let ((lhs (nb:make-aval '(2 3) :f32)) (rhs (nb:make-aval '(3 2) :f32)))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list lhs rhs) :lhs-contracting '(1 . 2) :rhs-contracting '(0)
                         :lhs-batch '() :rhs-batch '()))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list lhs rhs) :lhs-contracting '(1) :rhs-contracting '(a)
                         :lhs-batch '() :rhs-batch '()))
    (signals nb:primitive-error
      (%abstract-eval-of :dot-general (list lhs rhs) :lhs-contracting '(1) :rhs-contracting '(0)
                         :lhs-batch 5 :rhs-batch '()))))

;;; ============================== emit ==============================

(test dot-general/emit-matches-fixture
  "batch 無しの dot-general の :emit（f32・bf16）は、SSA 名を正規化した後
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
               (%normalize-ssa-names
                (format nil "~{~A~^~%~}" (%fixture-op-lines "dot_general_bf16"))))))

(test dot-general/emit-accumulates-bf16-f16-in-f32
  "bf16 / f16 の :emit は、f32 の結果型を持つ stablehlo.dot_general と、
それを元の dtype に戻す stablehlo.convert の2行を返す（issue #54）。
f32 / f64 は今まで通り1行のまま変わらない。"
  (dolist (dtype '(:bf16 :f16))
    (let* ((text (%emit-of :dot-general '("%a" "%b")
                            (list (nb:make-aval '(4 64) dtype) (nb:make-aval '(64 4) dtype))
                            "%12" (nb:make-aval '(4 4) dtype)
                            :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))
           (lines (uiop:split-string text :separator '(#\Newline))))
      (is (= 2 (length lines)) "~S dtype: expected 2 lines, got ~S" dtype lines)
      (is (string= "%acc_12" (subseq (first lines) 0 7)))
      (is (search "-> tensor<4x4xf32>" (first lines)))
      (is (string= "%12 = stablehlo.convert %acc_12 : (tensor<4x4xf32>) -> "
                   (subseq (second lines) 0 (length "%12 = stablehlo.convert %acc_12 : (tensor<4x4xf32>) -> "))))))
  (dolist (dtype '(:f32 :f64))
    (let ((text (%emit-of :dot-general '("%a" "%b")
                           (list (nb:make-aval '(4 64) dtype) (nb:make-aval '(64 4) dtype))
                           "%12" (nb:make-aval '(4 4) dtype)
                           :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '())))
      (is (= 1 (length (uiop:split-string text :separator '(#\Newline))))
          "~S dtype should stay a single line" dtype))))

(test dot-general/emit-with-batch-and-empty-contracting
  "batch 付きの :emit は batching_dims と contracting_dims の両方を出す。
縮約次元が空の :emit は contracting_dims = [] x [] を出し、batch が空なら
batching_dims 自体を省く。"
  (is (string= "%0 = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>"
               (%emit-of :dot-general '("%a" "%b")
                         (list (nb:make-aval '(2 3 4) :f32) (nb:make-aval '(2 4 5) :f32))
                         "%0" (nb:make-aval '(2 3 5) :f32)
                         :lhs-contracting '(2) :rhs-contracting '(1) :lhs-batch '(0) :rhs-batch '(0))))
  (is (string= "%0 = stablehlo.dot_general %a, %b, contracting_dims = [] x [] : (tensor<2x3xf32>, tensor<4x5xf32>) -> tensor<2x3x4x5xf32>"
               (%emit-of :dot-general '("%a" "%b")
                         (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(4 5) :f32))
                         "%0" (nb:make-aval '(2 3 4 5) :f32)
                         :lhs-contracting '() :rhs-contracting '() :lhs-batch '() :rhs-batch '()))))

;;; ===================== K=0（issue #62）: ゼロ定数 :emit =====================
;;;
;;; IREE 3.11.0 のコンパイラは、縮約次元（contracting dim）のサイズが0の
;;; dot_general で AnnotateDispatches の整数0除算により落ちる（issue #62）。
;;; :emit はこの形のとき dot_general を出さず、数学的に正しい結果
;;; （空和 = 0）であるゼロ定数を1行だけ出す。

(test dot-general/emit-zero-constant-for-zero-contracting-f32
  "K=0（lhs (2 0)、rhs (0 3)、contracting [1] x [0]）の :emit（f32）は、
dot_general ではなく `stablehlo.constant dense<0.0>` を1行だけ出す。"
  (is (string= "%0 = stablehlo.constant dense<0.0> : tensor<2x3xf32>"
               (%emit-of :dot-general '("%a" "%b")
                         (list (nb:make-aval '(2 0) :f32) (nb:make-aval '(0 3) :f32))
                         "%0" (nb:make-aval '(2 3) :f32)
                         :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))))

(test dot-general/emit-zero-constant-for-zero-contracting-bf16-f16
  "K=0 の :emit は bf16 / f16 でも f32 累積の acc/convert を経由せず、
ゼロ定数1行だけを出す（issue #54 の分岐より前で処理する）。"
  (dolist (dtype '(:bf16 :f16))
    (let ((text (%emit-of :dot-general '("%a" "%b")
                           (list (nb:make-aval '(2 0) dtype) (nb:make-aval '(0 3) dtype))
                           "%0" (nb:make-aval '(2 3) dtype)
                           :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '())))
      (is (string= (format nil "%0 = stablehlo.constant dense<0.0> : tensor<2x3x~(~A~)>" dtype) text))
      (is (not (search "%acc_" text)))
      (is (not (search "stablehlo.convert" text))))))

(test dot-general/emit-k1-still-emits-dot-general
  "K=1（境界値。ゼロサイズではない）は今まで通り stablehlo.dot_general を
出す（K=0 判定が < や <= ではなく = 0 であることを確かめる）。"
  (is (search "stablehlo.dot_general"
              (%emit-of :dot-general '("%a" "%b")
                        (list (nb:make-aval '(2 1) :f32) (nb:make-aval '(1 3) :f32))
                        "%0" (nb:make-aval '(2 3) :f32)
                        :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))))

(test dot-general/emit-zero-constant-batched
  "batch 付き K=0（lhs (2 3 0)、rhs (2 0 5)、batch (0)x(0)、
contracting (2)x(1)）の :emit もゼロ定数1行になる。"
  (is (string= "%0 = stablehlo.constant dense<0.0> : tensor<2x3x5xf32>"
               (%emit-of :dot-general '("%a" "%b")
                         (list (nb:make-aval '(2 3 0) :f32) (nb:make-aval '(2 0 5) :f32))
                         "%0" (nb:make-aval '(2 3 5) :f32)
                         :lhs-contracting '(2) :rhs-contracting '(1) :lhs-batch '(0) :rhs-batch '(0)))))

(test dot-general/emit-zero-constant-only-second-contracting-dim-zero
  "縮約次元が2つあり、後ろの次元だけがゼロサイズ（lhs (2 3 0)、
rhs (3 0 4)、contracting (1 2) x (0 1)）でも判定される
（K=0 判定が最初の縮約次元だけを見ていないことを確かめる）。"
  (is (string= "%0 = stablehlo.constant dense<0.0> : tensor<2x4xf32>"
               (%emit-of :dot-general '("%a" "%b")
                         (list (nb:make-aval '(2 3 0) :f32) (nb:make-aval '(3 0 4) :f32))
                         "%0" (nb:make-aval '(2 4) :f32)
                         :lhs-contracting '(1 2) :rhs-contracting '(0 1) :lhs-batch '() :rhs-batch '()))))

(test dot-general/emit-zero-constant-rank0-and-zero-size-output
  "完全に縮約された rank0 の K=0（lhs (0)、rhs (0)、contracting (0)x(0)）は
`tensor<f32>` のゼロ定数、ゼロサイズ出力の K=0（lhs (0 0)、rhs (0 3)）は
`tensor<0x3xf32>` のゼロ定数になる（出力側の特殊扱いは不要）。"
  (is (string= "%0 = stablehlo.constant dense<0.0> : tensor<f32>"
               (%emit-of :dot-general '("%a" "%b")
                         (list (nb:make-aval '(0) :f32) (nb:make-aval '(0) :f32))
                         "%0" (nb:make-aval '() :f32)
                         :lhs-contracting '(0) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '())))
  (is (string= "%0 = stablehlo.constant dense<0.0> : tensor<0x3xf32>"
               (%emit-of :dot-general '("%a" "%b")
                         (list (nb:make-aval '(0 0) :f32) (nb:make-aval '(0 3) :f32))
                         "%0" (nb:make-aval '(0 3) :f32)
                         :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))))

(test dot-general/eager-zero-contracting-returns-zeros
  "K=0 の eager 実装は、すべての要素が0の (2 3) 配列（f32）を返す
（:emit の変更は eager には影響しないことの回帰テスト）。"
  (let* ((lhs (make-array '(2 0) :element-type 'single-float))
         (rhs (make-array '(0 3) :element-type 'single-float))
         (result (%eager-of :dot-general (list lhs rhs)
                             (list (nb:array-aval lhs :f32) (nb:array-aval rhs :f32))
                             :lhs-contracting '(1) :rhs-contracting '(0)
                             :lhs-batch '() :rhs-batch '())))
    (is (equalp (nb:make-aval '(2 3) :f32) (nb:array-aval result :f32)))
    (is (loop for i below 2 always (loop for j below 3 always (zerop (aref result i j)))))))

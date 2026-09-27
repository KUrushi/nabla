;;;; reshape / broadcast-in-dim / transpose の性質（issue #31 p4）。
;;;;
;;;; 3つとも値を解釈せず raw storage を並べ替えるだけなので、:F32 :F64
;;;; :BF16 :F16 に加えて :I1（bit）でも同じように動くことを確かめる
;;;; （*SHAPE-TEST-DTYPES*）。find-primitive / primitive-abstract-eval /
;;;; primitive-emit / primitive-eager は内部シンボル（nb::）で呼ぶ（他の
;;;; primitive テストと同じ流儀。tests/primitive-test.lisp 参照）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defparameter *shape-test-dtypes* (list* :i1 *dtypes*)
  "reshape / broadcast-in-dim / transpose のテストで使う dtype の一覧。
raw storage をそのまま扱う演算なので、通常の4つの float dtype に加えて
:I1 も含める（契約 §4 のピットフォール1）。")

;;; --- 特定の値を毎回返す check-it generator（乱数を使う一発計算を
;;; check-it の generator プロトコルに乗せるための、薄いラッパー） ---

(defclass %const-generator (check-it:generator)
  ((value :initarg :value :reader %const-generator-value)))

(defmethod check-it:generate ((generator %const-generator))
  (%const-generator-value generator))

(defmethod check-it:shrink ((generator %const-generator) test)
  (declare (ignore test))
  (%const-generator-value generator))

(defun %const (value)
  (make-instance '%const-generator :value value))

(defun %ad-hoc-generator (thunk)
  "THUNK（引数無しの関数）を GENERATE のたびに呼んで、その返り値を生成する
check-it generator を返す。THUNK の中の CL:RANDOM 呼び出しは check-it が
束縛する *RANDOM-STATE* の下で評価されるので、他の check-it 組み込み
generator と同じく再現性・regression の記録の対象になる（SHRINK 自体は
恒等 = 縮小しない。UNIFORM-REAL と同じ考え方）。"
  (make-instance 'check-it:chained-generator
                 :pre-generators '()
                 :generator-function (lambda () (%const (funcall thunk)))))

;;; --- 乱数によるケース生成（shape / perm / dims の組を作る） ---

(defun %shuffled (list)
  "LIST の要素をランダムに並べ替えたリストを返す（Fisher-Yates）。"
  (let ((v (coerce list 'vector)))
    (loop for i from (1- (length v)) downto 1
          do (rotatef (aref v i) (aref v (random (1+ i)))))
    (coerce v 'list)))

(defun %distinct-dims (rank &key (max 8))
  "1..MAX から相異なる RANK 個の次元を選ぶ。mutation testing で
index 計算（stride の +/* の入れ替えなど）の変異を殺すには、次元が
すべて異なっていないと同じ結果になってしまうケースが多い（契約 §4 の
mutation の項）。"
  (subseq (%shuffled (loop for d from 1 to max collect d)) 0 rank))

(defun %random-factors (&key (max-count 4) (max-factor 4))
  (loop repeat (random (1+ max-count)) collect (1+ (random max-factor))))

(defun %scatter-into-shape (factors rank)
  "FACTORS（正整数のリスト）をランダムに RANK 個のバケツに配り、各バケツの
積を shape の各次元にする。RANK が0のときは FACTORS が空（積1）のときだけ
'() を返し、そうでなければ NIL（呼び出し側で作り直す）を返す。"
  (cond
    ((and (zerop rank) (null factors)) '())
    ((zerop rank) nil)
    (t (let ((buckets (make-array rank :initial-element 1)))
         (dolist (f factors)
           (let ((idx (random rank)))
             (setf (aref buckets idx) (* (aref buckets idx) f))))
         (coerce buckets 'list)))))

(defun %random-reshape-case ()
  "同じ要素数を持つ (in-shape out-shape) を返す。rank 0〜4、size 0 の次元も
出ることがある（FACTORS が空なら常に size 1 で、rank 0 の組も出る）。"
  (loop
    (let* ((factors (%random-factors))
           (in-shape (%scatter-into-shape factors (random 5)))
           (out-shape (%scatter-into-shape factors (random 5))))
      (when (and in-shape out-shape)
        (return (list in-shape out-shape))))))

(defun %random-transpose-case ()
  "(shape perm) を返す。shape の次元はすべて相異なる（mutation 対策）。"
  (let* ((rank (random 5))
         (shape (%distinct-dims rank)))
    (list shape (%shuffled (loop for i below rank collect i)))))

(defun %random-broadcast-case ()
  "(in-shape out-shape dims) を返す。out-shape の次元はすべて相異なり、
in-shape の各次元は1（broadcast する）か対応する out-shape の次元と同じ
値のどちらか、dims は out-shape の中からランダムに選んだ相異なる添字
（昇順とは限らない。StableHLO の broadcast_in_dim は unique であれば
よく、increasing は要求しない——契約のピットフォール(2)）。"
  (let* ((out-rank (random 5))
         (out-shape (%distinct-dims out-rank :max 9))
         (in-rank (random (1+ out-rank)))
         (dims (subseq (%shuffled (loop for i below out-rank collect i)) 0 in-rank))
         (in-shape (loop for d in dims collect (if (zerop (random 2)) 1 (nth d out-shape)))))
    (list in-shape out-shape dims)))

(defun %reshape-generator () (%ad-hoc-generator #'%random-reshape-case))
(defun %transpose-generator () (%ad-hoc-generator #'%random-transpose-case))
(defun %broadcast-generator () (%ad-hoc-generator #'%random-broadcast-case))

;;; --- primitive の呼び出しを短く書くためのヘルパー ---

(defun %abstract-eval-of (name in-avals &rest params)
  (apply (nb::primitive-abstract-eval (nb::find-primitive name)) in-avals params))

(defun %eager-of (name arrays in-avals &rest params)
  (apply (nb::primitive-eager (nb::find-primitive name)) arrays in-avals params))

(defun %emit-of (name in-names in-avals out-name out-aval &rest params)
  (apply (nb::primitive-emit (nb::find-primitive name)) in-names in-avals out-name out-aval params))

;;; --- 独立に書いた添字オラクル（%shape-row-major-index / %shape-subscripts
;;; とは別の計算方法。両方が同じバグを持っていたら見逃すので、実装本体とは
;;; 別のやり方——再帰的な divmod——で書く） ---

(defun %oracle-subscripts (index shape)
  (if (null shape)
      '()
      (let ((tail-size (reduce #'* (rest shape) :initial-value 1)))
        (cons (floor index tail-size) (%oracle-subscripts (mod index tail-size) (rest shape))))))

;;; --- 正規化して比較するための、フィクスチャの op 行の取り出し ---

(defun %shape-fixture-text (op-name)
  (let ((path (asdf:system-relative-pathname
               "nabla" (format nil "tests/fixtures/stablehlo/ops/~A.mlir" op-name))))
    (with-open-file (stream path :direction :input)
      (let ((text (make-string (file-length stream))))
        (subseq text 0 (read-sequence text stream))))))

(defun %fixture-op-lines (op-name)
  "tests/fixtures/stablehlo/ops/OP-NAME.mlir の func.func 行と func.return 行の
間にある行を、前後の空白を落として返す。"
  (let* ((lines (uiop:split-string (%shape-fixture-text op-name) :separator '(#\Newline)))
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

;;; ============================== reshape ==============================

(test shape/reshape/aval-matches-eager
  "reshape の abstract-eval が返す aval は、eager を実際に実行した結果の
aval（rank 0・size 0 の次元を含む）と、すべての dtype（:i1 含む）で一致する。"
  (is (check-it (%reshape-generator)
                (lambda (case)
                  (destructuring-bind (in-shape out-shape) case
                    (every (lambda (dtype)
                             (let* ((in (make-random-array (make-array-spec in-shape dtype)))
                                    (in-aval (nb:array-aval in dtype))
                                    (expected-aval (%abstract-eval-of :reshape (list in-aval) :shape out-shape))
                                    (result (%eager-of :reshape (list in) (list in-aval) :shape out-shape)))
                               (equalp expected-aval (nb:array-aval result (nb:aval-dtype expected-aval)))))
                           *shape-test-dtypes*)))
                :regression-id shape/reshape/aval-matches-eager
                :regression-file (regression-path "shape-reshape-aval"))))

(test shape/reshape/eager-matches-reference
  "reshape の eager 実装を decode-array で double-float に戻した結果は、
reference-reshape の期待値と dtype ごとの許容誤差で一致する。"
  (is (check-it (%reshape-generator)
                (lambda (case)
                  (destructuring-bind (in-shape out-shape) case
                    (every (lambda (dtype)
                             (let* ((in (make-random-array (make-array-spec in-shape dtype)))
                                    (in-aval (nb:array-aval in dtype))
                                    (result (%eager-of :reshape (list in) (list in-aval) :shape out-shape))
                                    (expected (reference-reshape (decode-array in dtype) out-shape)))
                               (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                 (allclose (decode-array result dtype) expected :rtol rtol :atol atol))))
                           *shape-test-dtypes*)))
                :regression-id shape/reshape/eager-matches-reference
                :regression-file (regression-path "shape-reshape-eager"))))

(test shape/reshape/round-trips-to-original-shape
  "reshape で別の shape にしてから元の shape に reshape し直すと、元の配列と
EQUALP で一致する（往復）。"
  (is (check-it (%reshape-generator)
                (lambda (case)
                  (destructuring-bind (in-shape out-shape) case
                    (let* ((in (make-random-array (make-array-spec in-shape :f32)))
                           (in-aval (nb:array-aval in :f32))
                           (mid (%eager-of :reshape (list in) (list in-aval) :shape out-shape))
                           (mid-aval (nb:array-aval mid :f32))
                           (back (%eager-of :reshape (list mid) (list mid-aval) :shape in-shape)))
                      (equalp in back))))
                :regression-id shape/reshape/round-trips-to-original-shape
                :regression-file (regression-path "shape-reshape-roundtrip"))))

(test shape/reshape/rank0-and-size0-examples
  "契約のピットフォール(3)(4): rank 0 <-> (1) / (1 1)、size 0 の次元
((0 3) <-> (0)) が primitive-error にならず、期待した aval になる。"
  (is (equalp (nb:make-aval '(1) :f32)
              (%abstract-eval-of :reshape (list (nb:make-aval '() :f32)) :shape '(1))))
  (is (equalp (nb:make-aval '(1 1) :f32)
              (%abstract-eval-of :reshape (list (nb:make-aval '() :f32)) :shape '(1 1))))
  (is (equalp (nb:make-aval '(0) :f32)
              (%abstract-eval-of :reshape (list (nb:make-aval '(0 3) :f32)) :shape '(0))))
  (let* ((scalar (make-random-array (make-array-spec '() :f32)))
         (result (%eager-of :reshape (list scalar) (list (nb:array-aval scalar :f32)) :shape '(1 1))))
    (is (equalp '(1 1) (array-dimensions result)))
    (is (= (aref scalar) (aref result 0 0)))))

(test shape/reshape/invalid-shape-signals-primitive-error
  "要素数が1つ違う shape、非リストの shape、負の次元を含む shape は
PRIMITIVE-ERROR になる。"
  (let ((in-aval (nb:make-aval '(2 3) :f32)))
    (signals nb:primitive-error (%abstract-eval-of :reshape (list in-aval) :shape '(3 3)))
    (signals nb:primitive-error (%abstract-eval-of :reshape (list in-aval) :shape 6))
    (signals nb:primitive-error (%abstract-eval-of :reshape (list in-aval) :shape '(2 -3)))))

(test shape/reshape/wrong-arity-signals-primitive-error
  "入力が0個・2個の reshape は PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error (%abstract-eval-of :reshape '() :shape '(1)))
  (signals nb:primitive-error
    (%abstract-eval-of :reshape (list (nb:make-aval '(2) :f32) (nb:make-aval '(2) :f32)) :shape '(2))))

(test shape/reshape/emit-matches-fixture
  "reshape の :emit（f32・bf16）は、SSA 名を正規化した後 reshape.mlir /
reshape_bf16.mlir の op 行と一致する。"
  (is (string= (%normalize-ssa-names
                (%emit-of :reshape '("%a") (list (nb:make-aval '(2 3) :f32)) "%0" (nb:make-aval '(3 2) :f32) :shape '(3 2)))
               (%normalize-ssa-names (first (%fixture-op-lines "reshape")))))
  (is (string= (%normalize-ssa-names
                (%emit-of :reshape '("%a") (list (nb:make-aval '(2 3) :bf16)) "%0" (nb:make-aval '(3 2) :bf16) :shape '(3 2)))
               (%normalize-ssa-names (first (%fixture-op-lines "reshape_bf16"))))))

;;; ========================= broadcast-in-dim =========================

(test shape/broadcast-in-dim/aval-matches-eager
  "broadcast-in-dim の abstract-eval と eager が返す aval が、すべての
dtype（:i1 含む）で一致する。"
  (is (check-it (%broadcast-generator)
                (lambda (case)
                  (destructuring-bind (in-shape out-shape dims) case
                    (every (lambda (dtype)
                             (let* ((in (make-random-array (make-array-spec in-shape dtype)))
                                    (in-aval (nb:array-aval in dtype))
                                    (expected-aval (%abstract-eval-of :broadcast-in-dim (list in-aval)
                                                                       :shape out-shape :dims dims))
                                    (result (%eager-of :broadcast-in-dim (list in) (list in-aval)
                                                        :shape out-shape :dims dims)))
                               (equalp expected-aval (nb:array-aval result (nb:aval-dtype expected-aval)))))
                           *shape-test-dtypes*)))
                :regression-id shape/broadcast-in-dim/aval-matches-eager
                :regression-file (regression-path "shape-broadcast-aval"))))

(test shape/broadcast-in-dim/eager-matches-reference
  "broadcast-in-dim の eager 実装を decode-array で戻した結果は、
reference-broadcast-in-dim の期待値と dtype ごとの許容誤差で一致する。"
  (is (check-it (%broadcast-generator)
                (lambda (case)
                  (destructuring-bind (in-shape out-shape dims) case
                    (every (lambda (dtype)
                             (let* ((in (make-random-array (make-array-spec in-shape dtype)))
                                    (in-aval (nb:array-aval in dtype))
                                    (result (%eager-of :broadcast-in-dim (list in) (list in-aval)
                                                        :shape out-shape :dims dims))
                                    (expected (reference-broadcast-in-dim (decode-array in dtype) out-shape dims)))
                               (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                 (allclose (decode-array result dtype) expected :rtol rtol :atol atol))))
                           *shape-test-dtypes*)))
                :regression-id shape/broadcast-in-dim/eager-matches-reference
                :regression-file (regression-path "shape-broadcast-eager"))))

(test shape/broadcast-in-dim/gather-matches-independent-oracle
  "eager が出力の各要素に書き込む値は、実装本体とは別の方法
（%ORACLE-SUBSCRIPTS）で計算した入力側の添字で引いた値と raw storage の
まま（デコードせず）一致する。"
  (is (check-it (%broadcast-generator)
                (lambda (case)
                  (destructuring-bind (in-shape out-shape dims) case
                    (let* ((in (make-random-array (make-array-spec in-shape :f32)))
                           (in-aval (nb:array-aval in :f32))
                           (result (%eager-of :broadcast-in-dim (list in) (list in-aval)
                                               :shape out-shape :dims dims)))
                      (loop for i below (array-total-size result)
                            for out-subs = (%oracle-subscripts i out-shape)
                            for in-subs = (loop for operand-dim in in-shape
                                                 for target-dim in dims
                                                 collect (if (= operand-dim 1) 0 (nth target-dim out-subs)))
                            always (eql (row-major-aref result i) (apply #'aref in in-subs))))))
                :regression-id shape/broadcast-in-dim/gather-matches-independent-oracle
                :regression-file (regression-path "shape-broadcast-oracle"))))

(test shape/broadcast-in-dim/identity-and-scalar-examples
  "dims = iota・同じ shape なら恒等コピー。rank 0 から任意の shape への
broadcast は、全要素がスカラー値になる。"
  (let* ((in (make-random-array (make-array-spec '(2 3) :f32)))
         (result (%eager-of :broadcast-in-dim (list in) (list (nb:array-aval in :f32)) :shape '(2 3) :dims '(0 1))))
    (is (equalp in result)))
  (let* ((scalar (make-random-array (make-array-spec '() :f32)))
         (result (%eager-of :broadcast-in-dim (list scalar) (list (nb:array-aval scalar :f32)) :shape '(2 3) :dims '())))
    (dotimes (i (array-total-size result))
      (is (= (aref scalar) (row-major-aref result i))))))

(test shape/broadcast-in-dim/invalid-dims-signal-primitive-error
  "dims が出力の rank と同じ（境界外）、-1（境界外）、重複、operand の次元が
1でも一致でもない、のいずれも PRIMITIVE-ERROR になる。"
  (let ((in-aval (nb:make-aval '(3 3) :f32)))
    (signals nb:primitive-error (%abstract-eval-of :broadcast-in-dim (list in-aval) :shape '(2 3) :dims '(0 2)))
    (signals nb:primitive-error (%abstract-eval-of :broadcast-in-dim (list in-aval) :shape '(2 3) :dims '(0 -1)))
    (signals nb:primitive-error (%abstract-eval-of :broadcast-in-dim (list in-aval) :shape '(2 3) :dims '(0 0)))
    (signals nb:primitive-error (%abstract-eval-of :broadcast-in-dim (list (nb:make-aval '(4) :f32)) :shape '(3) :dims '(0)))))

(test shape/broadcast-in-dim/wrong-dims-length-signals-primitive-error
  "dims の長さが operand の rank と違えば PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error
    (%abstract-eval-of :broadcast-in-dim (list (nb:make-aval '(3) :f32)) :shape '(2 3) :dims '())))

(test shape/broadcast-in-dim/emit-matches-fixture
  "broadcast-in-dim の :emit（f32・bf16）は、SSA 名を正規化した後
broadcast_in_dim.mlir / broadcast_in_dim_bf16.mlir の op 行と一致する。"
  (is (string= (%normalize-ssa-names
                (%emit-of :broadcast-in-dim '("%a") (list (nb:make-aval '(3) :f32)) "%0" (nb:make-aval '(2 3) :f32)
                          :shape '(2 3) :dims '(1)))
               (%normalize-ssa-names (first (%fixture-op-lines "broadcast_in_dim")))))
  (is (string= (%normalize-ssa-names
                (%emit-of :broadcast-in-dim '("%a") (list (nb:make-aval '(3) :bf16)) "%0" (nb:make-aval '(2 3) :bf16)
                          :shape '(2 3) :dims '(1)))
               (%normalize-ssa-names (first (%fixture-op-lines "broadcast_in_dim_bf16"))))))

;;; ============================= transpose =============================

(test shape/transpose/aval-matches-eager
  "transpose の abstract-eval と eager が返す aval が、すべての dtype
（:i1 含む）で一致する。"
  (is (check-it (%transpose-generator)
                (lambda (case)
                  (destructuring-bind (shape perm) case
                    (every (lambda (dtype)
                             (let* ((in (make-random-array (make-array-spec shape dtype)))
                                    (in-aval (nb:array-aval in dtype))
                                    (expected-aval (%abstract-eval-of :transpose (list in-aval) :perm perm))
                                    (result (%eager-of :transpose (list in) (list in-aval) :perm perm)))
                               (equalp expected-aval (nb:array-aval result (nb:aval-dtype expected-aval)))))
                           *shape-test-dtypes*)))
                :regression-id shape/transpose/aval-matches-eager
                :regression-file (regression-path "shape-transpose-aval"))))

(test shape/transpose/eager-matches-reference
  "transpose の eager 実装を decode-array で戻した結果は、
reference-transpose の期待値と dtype ごとの許容誤差で一致する。"
  (is (check-it (%transpose-generator)
                (lambda (case)
                  (destructuring-bind (shape perm) case
                    (every (lambda (dtype)
                             (let* ((in (make-random-array (make-array-spec shape dtype)))
                                    (in-aval (nb:array-aval in dtype))
                                    (result (%eager-of :transpose (list in) (list in-aval) :perm perm))
                                    (expected (reference-transpose (decode-array in dtype) perm)))
                               (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                 (allclose (decode-array result dtype) expected :rtol rtol :atol atol))))
                           *shape-test-dtypes*)))
                :regression-id shape/transpose/eager-matches-reference
                :regression-file (regression-path "shape-transpose-eager"))))

(test shape/transpose/gather-matches-independent-oracle
  "eager が出力の各要素に書き込む値は、実装本体とは別の方法
（%ORACLE-SUBSCRIPTS）で計算した入力側の添字で引いた値と raw storage の
まま一致する。"
  (is (check-it (%transpose-generator)
                (lambda (case)
                  (destructuring-bind (shape perm) case
                    (let* ((in (make-random-array (make-array-spec shape :f32)))
                           (in-aval (nb:array-aval in :f32))
                           (result (%eager-of :transpose (list in) (list in-aval) :perm perm))
                           (out-shape (array-dimensions result)))
                      (loop for i below (array-total-size result)
                            for out-subs = (%oracle-subscripts i out-shape)
                            for in-subs = (let ((v (make-list (length shape))))
                                            (loop for p in perm for s in out-subs do (setf (nth p v) s))
                                            v)
                            always (eql (row-major-aref result i) (apply #'aref in in-subs))))))
                :regression-id shape/transpose/gather-matches-independent-oracle
                :regression-file (regression-path "shape-transpose-oracle"))))

(test shape/transpose/identity-and-double-inverse-examples
  "identity perm は恒等コピー。perm を適用してからその逆 perm を適用すると
元の配列に戻る（往復）。rank 0 の transpose（perm '()）も恒等。"
  (let* ((in (make-random-array (make-array-spec '(2 3 4) :f32)))
         (identity-result (%eager-of :transpose (list in) (list (nb:array-aval in :f32)) :perm '(0 1 2))))
    (is (equalp in identity-result)))
  (let* ((in (make-random-array (make-array-spec '(2 3 4) :f32)))
         (perm '(2 0 1))
         (inverse (loop with inv = (make-list (length perm))
                        for p in perm for k from 0
                        do (setf (nth p inv) k)
                        finally (return inv)))
         (mid (%eager-of :transpose (list in) (list (nb:array-aval in :f32)) :perm perm))
         (back (%eager-of :transpose (list mid) (list (nb:array-aval mid :f32)) :perm inverse)))
    (is (equalp in back)))
  (let* ((scalar (make-random-array (make-array-spec '() :f32)))
         (result (%eager-of :transpose (list scalar) (list (nb:array-aval scalar :f32)) :perm '())))
    (is (equalp scalar result))))

(test shape/transpose/invalid-perm-signals-primitive-error
  "重複を含む perm（'(0 0)）、rank と長さの合わない perm（rank 2 に '(0 2)）
は PRIMITIVE-ERROR になる。"
  (let ((in-aval (nb:make-aval '(2 2) :f32)))
    (signals nb:primitive-error (%abstract-eval-of :transpose (list in-aval) :perm '(0 0)))
    (signals nb:primitive-error (%abstract-eval-of :transpose (list in-aval) :perm '(0 2)))
    (signals nb:primitive-error (%abstract-eval-of :transpose (list in-aval) :perm '(0)))))

(test shape/transpose/emit-matches-fixture
  "transpose の :emit（f32・bf16）は、SSA 名を正規化した後 transpose.mlir /
transpose_bf16.mlir の op 行と一致する。"
  (is (string= (%normalize-ssa-names
                (%emit-of :transpose '("%a") (list (nb:make-aval '(2 3 4) :f32)) "%0" (nb:make-aval '(4 2 3) :f32)
                          :perm '(2 0 1)))
               (%normalize-ssa-names (first (%fixture-op-lines "transpose")))))
  (is (string= (%normalize-ssa-names
                (%emit-of :transpose '("%a") (list (nb:make-aval '(2 3 4) :bf16)) "%0" (nb:make-aval '(4 2 3) :bf16)
                          :perm '(2 0 1)))
               (%normalize-ssa-names (first (%fixture-op-lines "transpose_bf16"))))))

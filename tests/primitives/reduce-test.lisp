;;;; reduce-sum / reduce-max の性質（issue #31 p6）。
;;;;
;;;; %SHUFFLED / %DISTINCT-DIMS / %AD-HOC-GENERATOR / %ABSTRACT-EVAL-OF /
;;;; %EAGER-OF / %EMIT-OF / %FIXTURE-OP-LINES / %NORMALIZE-SSA-NAMES /
;;;; %ORACLE-SUBSCRIPTS は tests/primitives/shape-test.lisp（同じ
;;;; nabla.tests パッケージ、p4）で定義済みのものをそのまま再利用する
;;;; （チェーンBの中、同じテストシステムなので DAMP の重複を作らない）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; --- ケース生成: shape（次元がすべて相異なる）と、その中の非空・
;;; 重複無し・昇順の axes の組を作る ---

(defun %random-reduce-case (&key (max-rank 4) (max-dim 8) (max-size 64))
  "(SHAPE AXES) を返す。SHAPE は rank 1..MAX-RANK、各次元は 1..MAX-DIM の
相異なる整数（%DISTINCT-DIMS。mutation testing で axis の入れ替えバグを
見逃さないため）で、要素数の合計が MAX-SIZE を超えないものだけを返す
（f32 の単精度累積和は要素数が増えるほど誤差が積み重なる。契約の
ピットフォール(5)「f32 sums of up to 64 elements in [-1,1) stay within
1e-5 rtol」に合わせ、reference との比較にも安全な上限にしている）。"
  (loop
    (let* ((rank (1+ (random max-rank)))
           (shape (%distinct-dims rank :max max-dim))
           (num-axes (1+ (random rank)))
           (axes (sort (subseq (%shuffled (loop for i below rank collect i)) 0 num-axes) #'<)))
      (when (<= (reduce #'* shape :initial-value 1) max-size)
        (return (list shape axes))))))

(defun %reduce-generator () (%ad-hoc-generator #'%random-reduce-case))

;; reduce-max/result-is-an-element-of-slice-and-is-at-least-every-slice-element
;; は出力の各要素ごとに入力全体を舐める O(size^2) のオラクルなので、
;; 他の性質より小さい shape を使う。
(defun %reduce-small-generator ()
  (%ad-hoc-generator (lambda () (%random-reduce-case :max-rank 3 :max-dim 4))))

;;; ============================ 性質 ============================

(test reduce-sum/aval-matches-eager
  "abstract-eval が返す aval は、eager を実際に実行した結果の aval と、
すべての float dtype で一致する。"
  (is (check-it (%reduce-generator)
                (lambda (case)
                  (destructuring-bind (shape axes) case
                    (every (lambda (dtype)
                             (let* ((array (make-random-array (make-array-spec shape dtype)))
                                    (in-aval (nb:array-aval array dtype))
                                    (expected-aval (%abstract-eval-of :reduce-sum (list in-aval) :axes axes))
                                    (result (%eager-of :reduce-sum (list array) (list in-aval) :axes axes)))
                               (equalp expected-aval (nb:array-aval result (nb:aval-dtype expected-aval)))))
                           *dtypes*)))
                :regression-id reduce-sum/aval-matches-eager
                :regression-file (regression-path "reduce-sum-aval-matches-eager"))))

(test reduce-max/aval-matches-eager
  "abstract-eval が返す aval は、eager を実際に実行した結果の aval と、
すべての float dtype で一致する。"
  (is (check-it (%reduce-generator)
                (lambda (case)
                  (destructuring-bind (shape axes) case
                    (every (lambda (dtype)
                             (let* ((array (make-random-array (make-array-spec shape dtype)))
                                    (in-aval (nb:array-aval array dtype))
                                    (expected-aval (%abstract-eval-of :reduce-max (list in-aval) :axes axes))
                                    (result (%eager-of :reduce-max (list array) (list in-aval) :axes axes)))
                               (equalp expected-aval (nb:array-aval result (nb:aval-dtype expected-aval)))))
                           *dtypes*)))
                :regression-id reduce-max/aval-matches-eager
                :regression-file (regression-path "reduce-max-aval-matches-eager"))))

(test reduce-sum/eager-matches-reference
  "eager 実装を decode-array で double-float に戻した結果は、
reference-reduce-sum の期待値と dtype ごとの許容誤差で一致する（f32 の
総和は最大64要素・値域 [-1, 1) なので rtol 1e-5 に収まる。順序依存の
誤差は、この規模では出ない）。"
  (is (check-it (%reduce-generator)
                (lambda (case)
                  (destructuring-bind (shape axes) case
                    (every (lambda (dtype)
                             (let* ((array (make-random-array (make-array-spec shape dtype)))
                                    (in-aval (nb:array-aval array dtype))
                                    (result (%eager-of :reduce-sum (list array) (list in-aval) :axes axes))
                                    (expected (reference-reduce-sum (decode-array array dtype) axes)))
                               (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                 (allclose (decode-array result dtype) expected :rtol rtol :atol atol))))
                           *dtypes*)))
                :regression-id reduce-sum/eager-matches-reference
                :regression-file (regression-path "reduce-sum-eager-matches-reference"))))

(test reduce-max/eager-matches-reference
  "eager 実装を decode-array で double-float に戻した結果は、
reference-reduce-max の期待値と dtype ごとの許容誤差で一致する。"
  (is (check-it (%reduce-generator)
                (lambda (case)
                  (destructuring-bind (shape axes) case
                    (every (lambda (dtype)
                             (let* ((array (make-random-array (make-array-spec shape dtype)))
                                    (in-aval (nb:array-aval array dtype))
                                    (result (%eager-of :reduce-max (list array) (list in-aval) :axes axes))
                                    (expected (reference-reduce-max (decode-array array dtype) axes)))
                               (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                 (allclose (decode-array result dtype) expected :rtol rtol :atol atol))))
                           *dtypes*)))
                :regression-id reduce-max/eager-matches-reference
                :regression-file (regression-path "reduce-max-eager-matches-reference"))))

(test reduce-sum/all-axes-equals-sequential-single-axis-reduction
  "全軸をまとめて reduce-sum した結果は、先頭の軸を1つずつ rank 回
繰り返し reduce-sum した結果と f64 の許容誤差で一致する（総和は結合的
なので、まとめて潰す順序に依らない。init 定数や accumulator の変異を
広い形状で殺す）。"
  (is (check-it (%reduce-generator)
                (lambda (case)
                  (destructuring-bind (shape axes) case
                    (declare (ignore axes))
                    (let* ((rank (length shape))
                           (array (make-random-array (make-array-spec shape :f64)))
                           (all-axes (loop for i below rank collect i))
                           (at-once (%eager-of :reduce-sum (list array) (list (nb:array-aval array :f64)) :axes all-axes))
                           (sequential array))
                      (dotimes (k rank)
                        (setf sequential
                              (%eager-of :reduce-sum (list sequential) (list (nb:array-aval sequential :f64)) :axes '(0))))
                      (multiple-value-bind (rtol atol) (dtype-tolerance :f64)
                        (allclose at-once sequential :rtol rtol :atol atol)))))
                :regression-id reduce-sum/all-axes-equals-sequential-single-axis-reduction
                :regression-file (regression-path "reduce-sum-all-axes-equals-sequential"))))

(test reduce-sum/ones-array-equals-product-of-reduced-dimension-sizes
  "全要素が1の配列を reduce-sum すると、各出力要素は reduce した次元の
サイズの積になる（init 0→1 の変異、accumulator の + → - の変異を殺す）。"
  (is (check-it (%reduce-generator)
                (lambda (case)
                  (destructuring-bind (shape axes) case
                    (let* ((ones (make-array shape :element-type 'single-float :initial-element 1.0))
                           (in-aval (nb:array-aval ones :f32))
                           (result (%eager-of :reduce-sum (list ones) (list in-aval) :axes axes))
                           (expected (coerce (reduce #'* (mapcar (lambda (a) (nth a shape)) axes)
                                                      :initial-value 1)
                                              'single-float)))
                      (loop for i below (array-total-size result)
                            always (= expected (row-major-aref result i))))))
                :regression-id reduce-sum/ones-array-equals-product-of-reduced-dimension-sizes
                :regression-file (regression-path "reduce-sum-ones-equals-product"))))

(test reduce-max/result-is-an-element-of-slice-and-is-at-least-every-slice-element
  "reduce-max の各出力要素は、対応する入力スライス（reduce する軸以外の
添字が出力の添字に一致する入力要素の集合）のどれかの値と一致し、かつ
スライスの全要素以上である。独立に書いた %ORACLE-SUBSCRIPTS でスライスを
数え上げるオラクル（実装本体の index 計算とは別経路）。スライスが空
（reduce する次元のどれかのサイズが0）なら、そのオラクルには比較対象の
要素が無いので、代わりに init 値の -inf になっていることを確かめる
（契約のピットフォール(3)）。"
  (is (check-it (%reduce-small-generator)
                (lambda (case)
                  (destructuring-bind (shape axes) case
                    (let* ((array (make-random-array (make-array-spec shape :f32)))
                           (in-aval (nb:array-aval array :f32))
                           (result (%eager-of :reduce-max (list array) (list in-aval) :axes axes))
                           (out-shape (array-dimensions result)))
                      (loop for out-i below (array-total-size result)
                            always
                            (let ((out-subs (%oracle-subscripts out-i out-shape))
                                  (slice-values '()))
                              (dotimes (in-i (array-total-size array))
                                (let* ((in-subs (%oracle-subscripts in-i shape))
                                       (candidate (loop for s in in-subs for d from 0
                                                         unless (member d axes) collect s)))
                                  (when (equal candidate out-subs)
                                    (push (row-major-aref array in-i) slice-values))))
                              (let ((value (row-major-aref result out-i)))
                                (if (null slice-values)
                                    (and (sb-ext:float-infinity-p value) (minusp value))
                                    (and (member value slice-values :test #'=)
                                         (every (lambda (v) (<= v value)) slice-values)))))))))
                :regression-id reduce-max/result-is-an-element-of-slice-and-is-at-least-every-slice-element
                :regression-file (regression-path "reduce-max-result-is-slice-element"))))

;;; ======================= 例ベース: NaN・size 0 =======================

(test reduce-max/nan-in-slice-propagates-to-that-output-element-only
  "reduce する軸の中に NaN が1つでもあれば、その出力要素だけが NaN に
なる。NaN を含まない他のスライスの出力要素はそのまま最大値になる。"
  (let* ((array (make-array '(2 3) :element-type 'single-float
                             :initial-contents '((1.0 2.0 3.0) (4.0 5.0 6.0))))
         (nan (nb::%make-single-float #x7FC00000)))
    (setf (aref array 0 1) nan)
    (let* ((in-aval (nb:array-aval array :f32))
           (result (%eager-of :reduce-max (list array) (list in-aval) :axes '(1))))
      (is (sb-ext:float-nan-p (aref result 0)))
      (is (= 6.0 (aref result 1))))))

(test reduce-sum/size-zero-reduced-dim-yields-zero
  "reduce される次元のサイズが0なら、結果の各要素は0になる。"
  (let* ((in-aval (nb:make-aval '(0 3) :f32))
         (array (make-array '(0 3) :element-type 'single-float))
         (result (%eager-of :reduce-sum (list array) (list in-aval) :axes '(0))))
    (is (equal '(3) (array-dimensions result)))
    (dotimes (i 3) (is (= 0.0 (aref result i))))))

(test reduce-max/size-zero-reduced-dim-yields-negative-infinity
  "reduce される次元のサイズが0なら、結果の各要素は -inf になる
（encode-float16 は -inf を正しく 0xFF80 / 0xFC00 に変換するので、
bf16 / f16 でも成り立つ——契約のピットフォール(3)参照）。"
  (let* ((in-aval (nb:make-aval '(0 3) :f32))
         (array (make-array '(0 3) :element-type 'single-float))
         (result (%eager-of :reduce-max (list array) (list in-aval) :axes '(0))))
    (is (equal '(3) (array-dimensions result)))
    (dotimes (i 3)
      (is (sb-ext:float-infinity-p (aref result i)))
      (is (minusp (aref result i))))))

;;; ======================= 不正な入力・境界値 =======================

(test reduce-sum/wrong-arity-signals-primitive-error
  "入力が0個・2個の reduce-sum は PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error (%abstract-eval-of :reduce-sum '() :axes '(0)))
  (signals nb:primitive-error
    (%abstract-eval-of :reduce-sum (list (nb:make-aval '(4 8) :f32) (nb:make-aval '(4 8) :f32)) :axes '(0))))

(test reduce-max/wrong-arity-signals-primitive-error
  "入力が0個・2個の reduce-max は PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error (%abstract-eval-of :reduce-max '() :axes '(0)))
  (signals nb:primitive-error
    (%abstract-eval-of :reduce-max (list (nb:make-aval '(4 8) :f32) (nb:make-aval '(4 8) :f32)) :axes '(0))))

(test reduce-sum/non-float-dtype-signals-primitive-error
  "float でない dtype（:i1）の入力は PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error (%abstract-eval-of :reduce-sum (list (nb:make-aval '(4 8) :i1)) :axes '(0))))

(test reduce-max/non-float-dtype-signals-primitive-error
  "float でない dtype（:i1）の入力は PRIMITIVE-ERROR になる。"
  (signals nb:primitive-error (%abstract-eval-of :reduce-max (list (nb:make-aval '(4 8) :i1)) :axes '(0))))

(test reduce-sum/invalid-axes-signal-primitive-error
  "境界: axis = rank（範囲外）、axis = -1、降順、重複、空リスト、ドット対
（真正なリストでない）はすべて PRIMITIVE-ERROR になる（生の TYPE-ERROR を
逃さない）。"
  (let ((in-avals (list (nb:make-aval '(4 8) :f32))))
    (signals nb:primitive-error (%abstract-eval-of :reduce-sum in-avals :axes '(2)))
    (signals nb:primitive-error (%abstract-eval-of :reduce-sum in-avals :axes '(-1)))
    (signals nb:primitive-error (%abstract-eval-of :reduce-sum in-avals :axes '(1 0)))
    (signals nb:primitive-error (%abstract-eval-of :reduce-sum in-avals :axes '(0 0)))
    (signals nb:primitive-error (%abstract-eval-of :reduce-sum in-avals :axes '()))
    (signals nb:primitive-error (%abstract-eval-of :reduce-sum in-avals :axes '(0 . 1)))))

(test reduce-max/invalid-axes-signal-primitive-error
  "境界: axis = rank（範囲外）、axis = -1、降順、重複、空リスト、ドット対
（真正なリストでない）はすべて PRIMITIVE-ERROR になる（生の TYPE-ERROR を
逃さない）。"
  (let ((in-avals (list (nb:make-aval '(4 8) :f32))))
    (signals nb:primitive-error (%abstract-eval-of :reduce-max in-avals :axes '(2)))
    (signals nb:primitive-error (%abstract-eval-of :reduce-max in-avals :axes '(-1)))
    (signals nb:primitive-error (%abstract-eval-of :reduce-max in-avals :axes '(1 0)))
    (signals nb:primitive-error (%abstract-eval-of :reduce-max in-avals :axes '(0 0)))
    (signals nb:primitive-error (%abstract-eval-of :reduce-max in-avals :axes '()))
    (signals nb:primitive-error (%abstract-eval-of :reduce-max in-avals :axes '(0 . 1)))))

;;; ============================== emit ==============================

(defun %reduce-fixture-text (op-name)
  "OP-NAME.mlir の op 行（複数行）を、1つの文字列に改行区切りで戻す
（reduce の :emit は複数行を返すので、%FIXTURE-OP-LINES の各行を1行の
:emit 結果と同じ形に組み立て直す）。"
  (format nil "~{~A~^~%~}" (%fixture-op-lines op-name)))

(test reduce-sum/emit-matches-fixture
  "shape (4 8) → dimensions = [1] の :emit（f32・bf16）は、SSA 名を正規化
した後 reduce_add.mlir / reduce_add_bf16.mlir と一致する。"
  (is (string= (%normalize-ssa-names
                (%emit-of :reduce-sum '("%a") (list (nb:make-aval '(4 8) :f32))
                          "%0" (nb:make-aval '(4) :f32) :axes '(1)))
               (%normalize-ssa-names (%reduce-fixture-text "reduce_add"))))
  (is (string= (%normalize-ssa-names
                (%emit-of :reduce-sum '("%a") (list (nb:make-aval '(4 8) :bf16))
                          "%0" (nb:make-aval '(4) :bf16) :axes '(1)))
               (%normalize-ssa-names (%reduce-fixture-text "reduce_add_bf16")))))

(test reduce-max/emit-matches-fixture
  "shape (4 8) → dimensions = [1] の :emit（f32・bf16）は、SSA 名を正規化
した後 reduce_max.mlir / reduce_max_bf16.mlir と一致する。init は -inf の
16進ビット列で、\"dense<-inf>\" にはならない。"
  (is (string= (%normalize-ssa-names
                (%emit-of :reduce-max '("%a") (list (nb:make-aval '(4 8) :f32))
                          "%0" (nb:make-aval '(4) :f32) :axes '(1)))
               (%normalize-ssa-names (%reduce-fixture-text "reduce_max"))))
  (is (string= (%normalize-ssa-names
                (%emit-of :reduce-max '("%a") (list (nb:make-aval '(4 8) :bf16))
                          "%0" (nb:make-aval '(4) :bf16) :axes '(1)))
               (%normalize-ssa-names (%reduce-fixture-text "reduce_max_bf16")))))

(test reduce-sum/emit-reducing-all-axes-yields-rank0-scalar-type
  "全軸を reduce すると、出力の tensor 型は rank 0（要素の shape 無し）に
なる。"
  (is (string= (%emit-of :reduce-sum '("%a") (list (nb:make-aval '(4 8) :f32))
                          "%0" (nb:make-aval '() :f32) :axes '(0 1))
               "%init_0 = stablehlo.constant dense<0.0> : tensor<f32>
%0 = stablehlo.reduce(%a init: %init_0) applies stablehlo.add across dimensions = [0, 1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<f32>")))

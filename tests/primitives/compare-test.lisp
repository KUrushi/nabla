;;;; compare / select / convert の性質（issue #31 p3）。
;;;;
;;;; find-primitive / primitive-abstract-eval / primitive-emit /
;;;; primitive-eager は内部シンボル（nb::）で呼ぶ（契約 §0、tests/primitives/
;;;; arith-test.lisp と同じ方針）。ゴールデン emit のフィクスチャ読み込み
;;;; ヘルパーは、この unit 専用のコピーを持つ（契約 §4、DAMP 重複を許容）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; --- ゴールデン emit テスト用のフィクスチャ読み込みヘルパー ---

(defun %read-compare-op-fixture (name)
  "tests/fixtures/stablehlo/ops/NAME.mlir の内容を文字列で返す。"
  (let ((path (asdf:system-relative-pathname
               "nabla" (format nil "tests/fixtures/stablehlo/ops/~A.mlir" name))))
    (with-open-file (stream path :direction :input)
      (let ((text (make-string (file-length stream))))
        (subseq text 0 (read-sequence text stream))))))

(defun %split-compare-lines (text)
  (with-input-from-string (s text)
    (loop for line = (read-line s nil nil)
          while line collect line)))

(defun %compare-fixture-op-lines (name)
  "NAME フィクスチャの、func.func の行の次から func.return の行の前までの
行（前後の空白を trim したもの）のリストを返す。"
  (let* ((lines (%split-compare-lines (%read-compare-op-fixture name)))
         (start (position-if (lambda (l) (search "func.func" l)) lines))
         (end (position-if (lambda (l) (search "func.return" l)) lines)))
    (mapcar (lambda (l) (string-trim '(#\Space #\Tab) l))
            (subseq lines (1+ start) end))))

(defun %normalize-compare-ssa-names (text)
  "TEXT 中の各 %[A-Za-z0-9_]+ トークンを \"%\" に置き換える（SSA 名の違いを
無視して比べるため）。"
  (with-output-to-string (out)
    (let ((i 0) (n (length text)))
      (loop while (< i n) do
        (if (char= (char text i) #\%)
            (progn
              (write-char #\% out)
              (incf i)
              (loop while (and (< i n)
                               (or (alphanumericp (char text i)) (char= (char text i) #\_)))
                    do (incf i)))
            (progn (write-char (char text i) out) (incf i)))))))

;;; --- 生成ヘルパー ---

(defparameter %compare-directions '(:lt :le :gt :ge :eq :ne))

(defun %tie-array (a b-raw seed)
  "B-RAW と同じ形の新しい配列を返す。SEED から作った独立な乱数状態で
要素ごとに約半分の確率で A の値をコピーする（compare の :LE / :GE / :EQ が
:LT / :GT / :NE と異なる結果を出す入力を作るため。独立な乱数配列どうしでは
f32/f64 でほとんど値が一致せず、境界の性質を確かめられない）。"
  (let ((state (sb-ext:seed-random-state (logxor seed #xC0FFEE)))
        (result (make-array (array-dimensions a) :element-type (array-element-type a))))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i)
            (if (zerop (random 2 state)) (row-major-aref a i) (row-major-aref b-raw i))))))

(defun %bit-array (spec seed)
  "SPEC の shape・SEED から :i1 の乱数配列（BIT の配列）を作る。"
  (make-random-array (make-array-spec (array-spec-shape spec) :i1) :seed seed))

(defun %constant-bit-array (shape bit)
  (make-array shape :element-type 'bit :initial-element bit))

(defun %compare-nan-value (dtype)
  "DTYPE の格納表現を持つ quiet NaN の値を1つ返す（rank 0 配列に詰める用途）。
ビットパターンは NB::%QUIET-NAN（実装の1つの情報源）から取る
（tests/primitives/unary-test.lisp の %DTYPE-NAN-VALUE と同じ方針。DAMP
重複は契約 §4 で許容）。"
  (ecase dtype
    (:f32 (nb::%quiet-nan 'single-float))
    (:f64 (nb::%quiet-nan 'double-float))
    ((:bf16 :f16) (nb::encode-float16 (nb::%quiet-nan 'single-float) dtype))))

(defun %compare-scalar-array (dtype value)
  "DTYPE の格納表現で VALUE を1つだけ持つ rank 0 配列を返す。"
  (let ((array (make-array nil :element-type (nb:dtype-element-type dtype))))
    (setf (row-major-aref array 0) value)
    array))

(defun %compare-encode-value (double-value dtype)
  "DOUBLE-VALUE（DOUBLE-FLOAT）を DTYPE の格納表現に変換する
（:f32/:f64 はそのまま coerce、:bf16/:f16 は NB::ENCODE-FLOAT16 を経由）。"
  (ecase dtype
    (:f32 (coerce double-value 'single-float))
    (:f64 (coerce double-value 'double-float))
    ((:bf16 :f16) (nb::encode-float16 (coerce double-value 'single-float) dtype))))

;;; --- 性質1: aval(eager) = abstract-eval ---

(test primitives/compare/aval-matches-abstract-eval
  "compare の eager 実装の結果の aval は、abstract-eval が返す aval と一致する
（全 dtype・全 direction・rank 0〜4）。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (a (make-random-array spec :seed seed))
                           (b (make-random-array spec :seed (1+ seed)))
                           (prim (nb::find-primitive :compare))
                           (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype))))
                      (every (lambda (direction)
                               (let* ((out-aval (funcall (nb::primitive-abstract-eval prim) in-avals :direction direction))
                                      (result (funcall (nb::primitive-eager prim) (list a b) in-avals :direction direction)))
                                 (equalp (nb:array-aval result :i1) out-aval)))
                             %compare-directions))))
                :regression-id primitives/compare/aval-matches-abstract-eval
                :regression-file (regression-path "primitives-compare-compare-aval"))))

(test primitives/select/aval-matches-abstract-eval
  "select の eager 実装の結果の aval は、abstract-eval が返す aval と一致する
（全 dtype・rank 0〜4）。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (pred (%bit-array spec seed))
                           (a (make-random-array spec :seed (1+ seed)))
                           (b (make-random-array spec :seed (+ seed 2)))
                           (prim (nb::find-primitive :select))
                           (in-avals (list (nb:array-aval pred :i1) (nb:array-aval a dtype) (nb:array-aval b dtype)))
                           (out-aval (funcall (nb::primitive-abstract-eval prim) in-avals))
                           (result (funcall (nb::primitive-eager prim) (list pred a b) in-avals)))
                      (equalp (nb:array-aval result dtype) out-aval))))
                :regression-id primitives/select/aval-matches-abstract-eval
                :regression-file (regression-path "primitives-compare-select-aval"))))

(test primitives/convert/aval-matches-abstract-eval
  "convert の eager 実装の結果の aval は、abstract-eval が返す aval と一致する
（全 dtype の組み合わせ・rank 0〜4）。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*) (uniform-integer :lo 0 :hi 3)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec dtype-index seed) args
                    (let* ((in-dtype (array-spec-dtype spec))
                           (out-dtype (nth dtype-index *dtypes*))
                           (a (make-random-array spec :seed seed))
                           (prim (nb::find-primitive :convert))
                           (in-avals (list (nb:array-aval a in-dtype)))
                           (out-aval (funcall (nb::primitive-abstract-eval prim) in-avals :dtype out-dtype))
                           (result (funcall (nb::primitive-eager prim) (list a) in-avals :dtype out-dtype)))
                      (equalp (nb:array-aval result out-dtype) out-aval))))
                :regression-id primitives/convert/aval-matches-abstract-eval
                :regression-file (regression-path "primitives-compare-convert-aval"))))

;;; --- 性質2: eager = 参照実装 ---

(test primitives/compare/eager-matches-reference
  "compare の eager 実装の結果は REFERENCE-COMPARE と全 dtype・全 direction で
（ビット単位で厳密に）一致する。b は a と約半分の要素が等しくなるように作る
ので、:LT と :LE、:GT と :GE のような境界の違いも検出できる。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (a (make-random-array spec :seed seed))
                           (b-raw (make-random-array spec :seed (1+ seed)))
                           (b (%tie-array a b-raw seed))
                           (da (decode-array a dtype))
                           (db (decode-array b dtype))
                           (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype)))
                           (eager (nb::primitive-eager (nb::find-primitive :compare))))
                      (every (lambda (direction)
                               (equalp (funcall eager (list a b) in-avals :direction direction)
                                       (reference-compare da db direction)))
                             %compare-directions))))
                :regression-id primitives/compare/eager-matches-reference
                :regression-file (regression-path "primitives-compare-compare-reference"))))

(test primitives/select/eager-matches-reference
  "select の eager 実装の結果は、decode したうえで REFERENCE-SELECT と全
dtype で厳密に一致する（raw storage をそのままコピーするだけなので、
丸め誤差は入らない）。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (pred (%bit-array spec seed))
                           (a (make-random-array spec :seed (1+ seed)))
                           (b (make-random-array spec :seed (+ seed 2)))
                           (in-avals (list (nb:array-aval pred :i1) (nb:array-aval a dtype) (nb:array-aval b dtype)))
                           (result (funcall (nb::primitive-eager (nb::find-primitive :select))
                                             (list pred a b) in-avals)))
                      (equalp (decode-array result dtype)
                              (reference-select pred (decode-array a dtype) (decode-array b dtype))))))
                :regression-id primitives/select/eager-matches-reference
                :regression-file (regression-path "primitives-compare-select-reference"))))

(test primitives/convert/eager-matches-reference
  "convert(x, dtype) を decode したものは、x を decode したものと dtype の
許容誤差で一致する（値そのものは変えない演算なので、bf16/f16 が絡んでも
許容誤差の範囲に収まる）。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*) (uniform-integer :lo 0 :hi 3)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec dtype-index seed) args
                    (let* ((in-dtype (array-spec-dtype spec))
                           (out-dtype (nth dtype-index *dtypes*))
                           (a (make-random-array spec :seed seed))
                           (in-avals (list (nb:array-aval a in-dtype)))
                           (result (funcall (nb::primitive-eager (nb::find-primitive :convert))
                                             (list a) in-avals :dtype out-dtype)))
                      (multiple-value-bind (rtol atol) (dtype-tolerance out-dtype)
                        (allclose (decode-array result out-dtype) (decode-array a in-dtype)
                                  :rtol rtol :atol atol)))))
                :regression-id primitives/convert/eager-matches-reference
                :regression-file (regression-path "primitives-compare-convert-reference"))))

;;; --- 性質3: 不正な入力は PRIMITIVE-ERROR ---

(test primitives/compare/invalid-inputs-signal-primitive-error
  "compare は: 2個以外の入力の個数、shape 不一致、dtype 不一致、非浮動小数点
dtype、未知の direction のどれでも PRIMITIVE-ERROR になる。"
  (let ((eval (nb::primitive-abstract-eval (nb::find-primitive :compare))))
    (signals nb:primitive-error (funcall eval '() :direction :lt) "0個の入力")
    (signals nb:primitive-error (funcall eval (list (nb:make-aval '(2 3) :f32)) :direction :lt) "1個の入力")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32))
                 :direction :lt)
      "3個の入力")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(3 2) :f32)) :direction :lt)
      "shape 不一致")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f64)) :direction :lt)
      "dtype 不一致")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(2 3) :i1) (nb:make-aval '(2 3) :i1)) :direction :eq)
      "非浮動小数点 dtype")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)) :direction :foo)
      "未知の direction")))

(test primitives/select/invalid-inputs-signal-primitive-error
  "select は: 3個以外の入力の個数、pred の dtype が :i1 でない、shape 不一致、
on-true/on-false の aval 不一致のどれでも PRIMITIVE-ERROR になる。"
  (let ((eval (nb::primitive-abstract-eval (nb::find-primitive :select))))
    (signals nb:primitive-error (funcall eval '()) "0個の入力")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(4) :i1) (nb:make-aval '(4) :f32)))
      "2個の入力")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(4) :i1) (nb:make-aval '(4) :f32)
                             (nb:make-aval '(4) :f32) (nb:make-aval '(4) :f32)))
      "4個の入力")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(4) :f32) (nb:make-aval '(4) :f32) (nb:make-aval '(4) :f32)))
      "pred の dtype が :i1 でない")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(4) :i1) (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32)))
      "pred と on-true/on-false の shape が違う")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(4) :i1) (nb:make-aval '(4) :f32) (nb:make-aval '(4) :f64)))
      "on-true と on-false の dtype が違う")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(4) :i1) (nb:make-aval '(4 1) :f32) (nb:make-aval '(4) :f32)))
      "on-true と on-false の shape が違う")))

(test primitives/convert/invalid-inputs-signal-primitive-error
  "convert は: 1個以外の入力の個数、入力が非浮動小数点 dtype、出力 dtype が
非浮動小数点（:i1）のどれでも PRIMITIVE-ERROR になる。"
  (let ((eval (nb::primitive-abstract-eval (nb::find-primitive :convert))))
    (signals nb:primitive-error (funcall eval '() :dtype :f32) "0個の入力")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(4) :f32) (nb:make-aval '(4) :f32)) :dtype :f32)
      "2個の入力")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(4) :i1)) :dtype :f32)
      "入力が :i1")
    (signals nb:primitive-error
        (funcall eval (list (nb:make-aval '(4) :f32)) :dtype :i1)
      "出力 dtype が :i1")))

;;; --- 性質4: golden emit テスト ---

(test primitives/compare/emit-matches-fixture
  "compare の emit は tests/fixtures/stablehlo/ops/compare.mlir /
compare_bf16.mlir の op 行と、SSA 名を正規化したうえで一致する（direction LT）。"
  (let ((prim (nb::find-primitive :compare)))
    (flet ((%check (fixture dtype)
             (let* ((aval (nb:make-aval '(4) dtype))
                    (out-aval (nb:make-aval '(4) :i1))
                    (emitted (funcall (nb::primitive-emit prim) '("%a" "%b") (list aval aval) "%0" out-aval
                                       :direction :lt))
                    (expected (first (%compare-fixture-op-lines fixture))))
               (is (string= (%normalize-compare-ssa-names emitted) (%normalize-compare-ssa-names expected))
                   "~A: got ~S, expected ~S" fixture emitted expected))))
      (%check "compare" :f32)
      (%check "compare_bf16" :bf16))))

(test primitives/select/emit-matches-fixture
  "select の emit は tests/fixtures/stablehlo/ops/select.mlir /
select_bf16.mlir の op 行と、SSA 名を正規化したうえで一致する。"
  (let ((prim (nb::find-primitive :select)))
    (flet ((%check (fixture dtype)
             (let* ((pred-aval (nb:make-aval '(4) :i1))
                    (aval (nb:make-aval '(4) dtype))
                    (emitted (funcall (nb::primitive-emit prim) '("%pred" "%a" "%b")
                                       (list pred-aval aval aval) "%0" aval))
                    (expected (first (%compare-fixture-op-lines fixture))))
               (is (string= (%normalize-compare-ssa-names emitted) (%normalize-compare-ssa-names expected))
                   "~A: got ~S, expected ~S" fixture emitted expected))))
      (%check "select" :f32)
      (%check "select_bf16" :bf16))))

(test primitives/convert/emit-matches-fixture
  "convert の emit は、f32→bf16 (convert.mlir) と bf16→f32 (convert_bf16.mlir)
のどちらも SSA 名を正規化したうえで一致する。"
  (let* ((prim (nb::find-primitive :convert))
         (f32-aval (nb:make-aval '(4) :f32))
         (bf16-aval (nb:make-aval '(4) :bf16)))
    (flet ((%check (fixture in-aval out-aval out-dtype)
             (let* ((emitted (funcall (nb::primitive-emit prim) '("%a") (list in-aval) "%0" out-aval :dtype out-dtype))
                    (expected (first (%compare-fixture-op-lines fixture))))
               (is (string= (%normalize-compare-ssa-names emitted) (%normalize-compare-ssa-names expected))
                   "~A: got ~S, expected ~S" fixture emitted expected))))
      (%check "convert" f32-aval bf16-aval :bf16)
      (%check "convert_bf16" bf16-aval f32-aval :f32))))

;;; --- 追加のオラクル（reference-* と実装を共有しないため、境界の変異を
;;; 独立に検出できる。mutation testing の生存対策） ---

(test primitives/compare/lt-is-flipped-gt
  "compare(a, b, :lt) は compare(b, a, :gt) と一致する（引数の順序を入れ替え
つつ方向も反転させると、結果は変わらない）。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (a (make-random-array spec :seed seed))
                           (b-raw (make-random-array spec :seed (1+ seed)))
                           (b (%tie-array a b-raw seed))
                           (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype)))
                           (eager (nb::primitive-eager (nb::find-primitive :compare))))
                      (equalp (funcall eager (list a b) in-avals :direction :lt)
                              (funcall eager (list b a) in-avals :direction :gt)))))
                :regression-id primitives/compare/lt-is-flipped-gt
                :regression-file (regression-path "primitives-compare-lt-is-flipped-gt"))))

(test primitives/compare/lt-is-not-ge
  "compare(a, b, :lt) は compare(a, b, :ge) の論理否定と一致する（NaN を含む
入力は対象外。:tie-array で一致する要素も混ぜているので、境界の LT/GE の
食い違いが実際に試される）。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (a (make-random-array spec :seed seed))
                           (b-raw (make-random-array spec :seed (1+ seed)))
                           (b (%tie-array a b-raw seed))
                           (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype)))
                           (eager (nb::primitive-eager (nb::find-primitive :compare)))
                           (lt (funcall eager (list a b) in-avals :direction :lt))
                           (ge (funcall eager (list a b) in-avals :direction :ge)))
                      (dotimes (i (array-total-size lt) t)
                        (unless (/= (row-major-aref lt i) (row-major-aref ge i))
                          (return nil))))))
                :regression-id primitives/compare/lt-is-not-ge
                :regression-file (regression-path "primitives-compare-lt-is-not-ge"))))

(test primitives/compare/nan-on-either-side
  "契約 §2 の NaN 規則: 比較の一方または両方が NaN のとき、:NE 以外の全方向
（:LT :LE :GT :GE :EQ）は 0（偽）を、:NE だけ 1（真）を返す（全 dtype、
NaN op 1.0 / 1.0 op NaN / NaN op NaN の3パターン）。"
  (dolist (dtype *dtypes*)
    (let* ((nan (%compare-scalar-array dtype (%compare-nan-value dtype)))
           (one (%compare-scalar-array dtype (%compare-encode-value 1.0d0 dtype)))
           (in-avals (list (nb:array-aval nan dtype) (nb:array-aval one dtype)))
           (eager (nb::primitive-eager (nb::find-primitive :compare))))
      (dolist (direction %compare-directions)
        (dolist (pair (list (list nan one) (list one nan) (list nan nan)))
          (let ((result (funcall eager pair in-avals :direction direction)))
            (is (= (if (eq direction :ne) 1 0) (row-major-aref result 0))
                "~A/~A: direction=~A の結果が ~A でない"
                dtype pair direction (if (eq direction :ne) 1 0))))))))

(test primitives/select/on-true-equals-on-false-returns-that-value
  "select(pred, a, a) は pred の値によらず a と一致する。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (pred (%bit-array spec seed))
                           (a (make-random-array spec :seed (1+ seed)))
                           (in-avals (list (nb:array-aval pred :i1) (nb:array-aval a dtype) (nb:array-aval a dtype)))
                           (result (funcall (nb::primitive-eager (nb::find-primitive :select)) (list pred a a) in-avals)))
                      (equalp result a))))
                :regression-id primitives/select/on-true-equals-on-false-returns-that-value
                :regression-file (regression-path "primitives-compare-select-on-true-equals-on-false"))))

(test primitives/select/all-ones-pred-is-on-true
  "pred の全要素が1なら select は on-true をそのまま返す。"
  (is (check-it (generator (array-spec :dtypes *dtypes*))
                (lambda (spec)
                  (let* ((dtype (array-spec-dtype spec))
                         (shape (array-spec-shape spec))
                         (pred (%constant-bit-array shape 1))
                         (a (make-random-array spec :seed 1))
                         (b (make-random-array spec :seed 2))
                         (in-avals (list (nb:array-aval pred :i1) (nb:array-aval a dtype) (nb:array-aval b dtype)))
                         (result (funcall (nb::primitive-eager (nb::find-primitive :select)) (list pred a b) in-avals)))
                    (equalp result a)))
                :regression-id primitives/select/all-ones-pred-is-on-true
                :regression-file (regression-path "primitives-compare-select-all-ones"))))

(test primitives/select/all-zeros-pred-is-on-false
  "pred の全要素が0なら select は on-false をそのまま返す。"
  (is (check-it (generator (array-spec :dtypes *dtypes*))
                (lambda (spec)
                  (let* ((dtype (array-spec-dtype spec))
                         (shape (array-spec-shape spec))
                         (pred (%constant-bit-array shape 0))
                         (a (make-random-array spec :seed 1))
                         (b (make-random-array spec :seed 2))
                         (in-avals (list (nb:array-aval pred :i1) (nb:array-aval a dtype) (nb:array-aval b dtype)))
                         (result (funcall (nb::primitive-eager (nb::find-primitive :select)) (list pred a b) in-avals)))
                    (equalp result b)))
                :regression-id primitives/select/all-zeros-pred-is-on-false
                :regression-file (regression-path "primitives-compare-select-all-zeros"))))

(test primitives/select/compare-lt-then-select-is-min
  "select(compare(a, b, :lt), a, b) は有限の入力について min(a, b)（decode
したうえで、丸め誤差なしで厳密に）と一致する。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (a (make-random-array spec :seed seed))
                           (b-raw (make-random-array spec :seed (1+ seed)))
                           (b (%tie-array a b-raw seed))
                           (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype)))
                           (pred (funcall (nb::primitive-eager (nb::find-primitive :compare)) (list a b) in-avals :direction :lt))
                           (select-in-avals (list (nb:array-aval pred :i1) (nb:array-aval a dtype) (nb:array-aval b dtype)))
                           (result (funcall (nb::primitive-eager (nb::find-primitive :select)) (list pred a b) select-in-avals))
                           (da (decode-array a dtype))
                           (db (decode-array b dtype))
                           (dr (decode-array result dtype)))
                      (dotimes (i (array-total-size dr) t)
                        (unless (= (row-major-aref dr i) (min (row-major-aref da i) (row-major-aref db i)))
                          (return nil))))))
                :regression-id primitives/select/compare-lt-then-select-is-min
                :regression-file (regression-path "primitives-compare-select-min"))))

(test primitives/convert/round-trip-through-f64-is-identity-for-f32
  "f32 の x について convert(convert(x, :f64), :f32) は x と厳密に（raw
storage のビットまで）一致する（f32 → f64 は正確な拡大、f64 → f32 で元の
値に戻る）。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f32)) (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((x (make-random-array spec :seed seed))
                           (in-avals (list (nb:array-aval x :f32)))
                           (widened (funcall (nb::primitive-eager (nb::find-primitive :convert)) (list x) in-avals :dtype :f64))
                           (narrowed (funcall (nb::primitive-eager (nb::find-primitive :convert))
                                               (list widened) (list (nb:array-aval widened :f64)) :dtype :f32)))
                      (equalp narrowed x))))
                :regression-id primitives/convert/round-trip-through-f64-is-identity-for-f32
                :regression-file (regression-path "primitives-compare-convert-f64-round-trip"))))

(test primitives/convert/same-dtype-is-a-fresh-equal-copy
  "convert(x, 同じ dtype) は x と equalp だが同一の配列オブジェクトではない
（新しい配列を作って返す）。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*) (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (x (make-random-array spec :seed seed))
                           (in-avals (list (nb:array-aval x dtype)))
                           (result (funcall (nb::primitive-eager (nb::find-primitive :convert)) (list x) in-avals :dtype dtype)))
                      (and (not (eq result x)) (equalp result x)))))
                :regression-id primitives/convert/same-dtype-is-a-fresh-equal-copy
                :regression-file (regression-path "primitives-compare-convert-same-dtype"))))

(test primitives/convert/f32-to-bf16-matches-encode-float16-array
  "convert(x, :bf16)（x は f32）は NB::ENCODE-FLOAT16-ARRAY :bf16 の結果と
ビット単位で一致する。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f32)) (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((x (make-random-array spec :seed seed))
                           (in-avals (list (nb:array-aval x :f32)))
                           (result (funcall (nb::primitive-eager (nb::find-primitive :convert)) (list x) in-avals :dtype :bf16)))
                      (equalp result (nb::encode-float16-array x :bf16)))))
                :regression-id primitives/convert/f32-to-bf16-matches-encode-float16-array
                :regression-file (regression-path "primitives-compare-convert-f32-to-bf16"))))

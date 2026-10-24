;;;; neg / exp / log / tanh / max / min の性質（issue #31 p2）。
;;;;
;;;; find-primitive / primitive-abstract-eval / primitive-emit /
;;;; primitive-eager は内部シンボル（nb::）で呼ぶ（契約 §0、tests/primitives/
;;;; arith-test.lisp と同じ方針）。ゴールデン emit のフィクスチャ読み込み
;;;; ヘルパーは、この unit 専用のコピーを持つ（契約 §4「各 unit が自分の
;;;; %fixture-op-lines / %normalize-ssa-names を持つ」、DAMP 重複を許容）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; --- ゴールデン emit テスト用のフィクスチャ読み込みヘルパー ---

(defun %read-unary-op-fixture (name)
  "tests/fixtures/stablehlo/ops/NAME.mlir の内容を文字列で返す。"
  (let ((path (asdf:system-relative-pathname
               "nabla" (format nil "tests/fixtures/stablehlo/ops/~A.mlir" name))))
    (with-open-file (stream path :direction :input)
      (let ((text (make-string (file-length stream))))
        (subseq text 0 (read-sequence text stream))))))

(defun %split-unary-lines (text)
  (with-input-from-string (s text)
    (loop for line = (read-line s nil nil)
          while line collect line)))

(defun %unary-fixture-op-lines (name)
  "NAME フィクスチャの、func.func の行の次から func.return の行の前までの
行（前後の空白を trim したもの）のリストを返す。"
  (let* ((lines (%split-unary-lines (%read-unary-op-fixture name)))
         (start (position-if (lambda (l) (search "func.func" l)) lines))
         (end (position-if (lambda (l) (search "func.return" l)) lines)))
    (mapcar (lambda (l) (string-trim '(#\Space #\Tab) l))
            (subseq lines (1+ start) end))))

(defun %normalize-unary-ssa-names (text)
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

;;; --- スカラー配列と decode（無限大・NaN を正しく扱う） ---

(defun %unary-scalar-array (dtype value)
  ;; ELEMENT-TYPE-FOR-DTYPE / DTYPE-VALUE は NABLA.TESTS.SUPPORT から
  ;; export されていない内部関数（tests/primitives/arith-test.lisp の
  ;; %scalar-array と同じ pitfall）。support 側のリファクタリングで
  ;; 名前や意味が変わってもここは静かに壊れうる。
  (let ((array (make-array '() :element-type (nabla.tests.support::element-type-for-dtype dtype))))
    (setf (row-major-aref array 0) (nabla.tests.support::dtype-value dtype value))
    array))

(defun %decode-unary-scalar (dtype value)
  "VALUE（DTYPE の格納表現を持つ1要素）を DOUBLE-FLOAT に戻す。bf16/f16 は
NB::DECODE-FLOAT16 を直接使う（DECODE-ELEMENT は無限大・NaN のビット
パターンを正しく扱わない既知の問題があるため。tests/primitives/arith-test.lisp
の %decode-scalar と同じ理由）。"
  (if (member dtype '(:bf16 :f16))
      (coerce (nb::decode-float16 value dtype) 'double-float)
      (coerce value 'double-float)))

(defun %decode-unary-array (array dtype)
  "ARRAY（DTYPE の格納表現）を %DECODE-UNARY-SCALAR で要素ごとにデコードした
DOUBLE-FLOAT の配列を返す。"
  (let ((result (make-array (array-dimensions array) :element-type 'double-float)))
    (dotimes (i (array-total-size array) result)
      (setf (row-major-aref result i) (%decode-unary-scalar dtype (row-major-aref array i))))))

(defun %unary-array-every (pred array)
  (dotimes (i (array-total-size array) t)
    (unless (funcall pred (row-major-aref array i))
      (return nil))))

(defun %dtype-nan-value (dtype)
  "DTYPE の格納表現を持つ quiet NaN の値を1つ返す（rank 0 配列に詰める用途）。
ビットパターンは NB::%QUIET-NAN（実装の1つの情報源）から取る。テストは
*値* が何らかの NaN であることだけを使うので、これで実装から独立性を
失うわけではない。"
  (ecase dtype
    (:f32 (nb::%quiet-nan 'single-float))
    (:f64 (nb::%quiet-nan 'double-float))
    ((:bf16 :f16) (nb::encode-float16 (nb::%quiet-nan 'single-float) dtype))))

(defun %unary-nan-array (dtype)
  "DTYPE の格納表現で quiet NaN を1つだけ持つ rank 0 配列を返す。"
  (let ((array (%unary-scalar-array dtype 0.0d0)))
    (setf (row-major-aref array 0) (%dtype-nan-value dtype))
    array))

;;; --- 性質1: aval(eager) = abstract-eval（単項） ---

(defmacro def-unary-aval-matches-abstract-eval-test (test-name prim-name)
  `(test ,test-name
     ,(format nil "~(~A~) の eager 実装の結果の aval は、abstract-eval が返す aval と一致する
（全 dtype・rank 0〜4）。" prim-name)
     (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                      (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                   (lambda (args)
                     (destructuring-bind (spec seed) args
                       (let* ((dtype (array-spec-dtype spec))
                              (a (make-random-array spec :seed seed))
                              (prim (nb::find-primitive ,prim-name))
                              (in-avals (list (nb:array-aval a dtype)))
                              (out-aval (funcall (nb::primitive-abstract-eval prim) in-avals))
                              (result (funcall (nb::primitive-eager prim) (list a) in-avals)))
                         (equalp (nb:array-aval result dtype) out-aval))))
                   :regression-id ,test-name
                   :regression-file (regression-path ,(format nil "primitives-unary-~(~A~)-aval" prim-name)))
         ,(format nil "~A: eager の結果の aval が abstract-eval と一致しなかった" prim-name))))

(def-unary-aval-matches-abstract-eval-test primitives/neg/aval-matches-abstract-eval :neg)
(def-unary-aval-matches-abstract-eval-test primitives/exp/aval-matches-abstract-eval :exp)
(def-unary-aval-matches-abstract-eval-test primitives/log/aval-matches-abstract-eval :log)
(def-unary-aval-matches-abstract-eval-test primitives/tanh/aval-matches-abstract-eval :tanh)
(def-unary-aval-matches-abstract-eval-test primitives/sqrt/aval-matches-abstract-eval :sqrt)
(def-unary-aval-matches-abstract-eval-test primitives/rsqrt/aval-matches-abstract-eval :rsqrt)
(def-unary-aval-matches-abstract-eval-test primitives/log1p/aval-matches-abstract-eval :log1p)
(def-unary-aval-matches-abstract-eval-test primitives/expm1/aval-matches-abstract-eval :expm1)

;;; --- 性質1': aval(eager) = abstract-eval（max / min。2入力） ---

(defmacro def-minmax-aval-matches-abstract-eval-test (test-name prim-name)
  `(test ,test-name
     ,(format nil "~(~A~) の eager 実装の結果の aval は、abstract-eval が返す aval と一致する
（全 dtype・rank 0〜4）。" prim-name)
     (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                      (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                   (lambda (args)
                     (destructuring-bind (spec seed) args
                       (let* ((dtype (array-spec-dtype spec))
                              (a (make-random-array spec :seed seed))
                              (b (make-random-array spec :seed (1+ seed)))
                              (prim (nb::find-primitive ,prim-name))
                              (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype)))
                              (out-aval (funcall (nb::primitive-abstract-eval prim) in-avals))
                              (result (funcall (nb::primitive-eager prim) (list a b) in-avals)))
                         (equalp (nb:array-aval result dtype) out-aval))))
                   :regression-id ,test-name
                   :regression-file (regression-path ,(format nil "primitives-minmax-~(~A~)-aval" prim-name)))
         ,(format nil "~A: eager の結果の aval が abstract-eval と一致しなかった" prim-name))))

(def-minmax-aval-matches-abstract-eval-test primitives/max/aval-matches-abstract-eval :max)
(def-minmax-aval-matches-abstract-eval-test primitives/min/aval-matches-abstract-eval :min)

;;; --- 性質2: eager = 参照実装（許容誤差つき） ---

(defmacro def-unary-eager-matches-reference-test (test-name prim-name reference-fn domain)
  `(test ,test-name
     ,(format nil "~(~A~) の eager 実装の結果は、%decode-unary-array で double-float に
戻したうえで ~A と全 dtype・rtol/atol で一致する。" prim-name reference-fn)
     (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                      (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                   (lambda (args)
                     (destructuring-bind (spec seed) args
                       (let* ((dtype (array-spec-dtype spec))
                              (a (make-random-array spec :seed seed :domain ,domain))
                              (in-avals (list (nb:array-aval a dtype)))
                              (result (funcall (nb::primitive-eager (nb::find-primitive ,prim-name))
                                                (list a) in-avals)))
                         (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                           (allclose (%decode-unary-array result dtype)
                                     (,reference-fn (%decode-unary-array a dtype))
                                     :rtol rtol :atol atol)))))
                   :regression-id ,test-name
                   :regression-file (regression-path ,(format nil "primitives-unary-~(~A~)-reference" prim-name))))))

(def-unary-eager-matches-reference-test primitives/neg/eager-matches-reference :neg reference-neg :any)
(def-unary-eager-matches-reference-test primitives/exp/eager-matches-reference :exp reference-exp :any)
(def-unary-eager-matches-reference-test primitives/log/eager-matches-reference :log reference-log :positive)
(def-unary-eager-matches-reference-test primitives/tanh/eager-matches-reference :tanh reference-tanh :any)
(def-unary-eager-matches-reference-test primitives/sqrt/eager-matches-reference :sqrt reference-sqrt :positive)
(def-unary-eager-matches-reference-test primitives/rsqrt/eager-matches-reference :rsqrt reference-rsqrt :positive)
(def-unary-eager-matches-reference-test primitives/log1p/eager-matches-reference :log1p reference-log1p :positive)
(def-unary-eager-matches-reference-test primitives/expm1/eager-matches-reference :expm1 reference-expm1 :any)

(defmacro def-minmax-eager-matches-reference-test (test-name prim-name reference-fn)
  `(test ,test-name
     ,(format nil "~(~A~) の eager 実装の結果は、%decode-unary-array で double-float に
戻したうえで ~A と全 dtype・rtol/atol で一致する。" prim-name reference-fn)
     (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                      (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                   (lambda (args)
                     (destructuring-bind (spec seed) args
                       (let* ((dtype (array-spec-dtype spec))
                              (a (make-random-array spec :seed seed))
                              (b (make-random-array spec :seed (1+ seed)))
                              (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype)))
                              (result (funcall (nb::primitive-eager (nb::find-primitive ,prim-name))
                                                (list a b) in-avals)))
                         (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                           (allclose (%decode-unary-array result dtype)
                                     (,reference-fn (%decode-unary-array a dtype) (%decode-unary-array b dtype))
                                     :rtol rtol :atol atol)))))
                   :regression-id ,test-name
                   :regression-file (regression-path ,(format nil "primitives-minmax-~(~A~)-reference" prim-name))))))

(def-minmax-eager-matches-reference-test primitives/max/eager-matches-reference :max reference-max)
(def-minmax-eager-matches-reference-test primitives/min/eager-matches-reference :min reference-min)

;;; --- 性質3: 不正な入力は PRIMITIVE-ERROR ---

(test primitives/unary/wrong-arity-signals-primitive-error
  "neg/exp/log/tanh/sqrt/rsqrt/log1p/expm1 はちょうど1つの入力を要求する。0個・2個ではどちらも
PRIMITIVE-ERROR になる。"
  (dolist (name '(:neg :exp :log :tanh :sqrt :rsqrt :log1p :expm1))
    (let ((eval (nb::primitive-abstract-eval (nb::find-primitive name))))
      (signals nb:primitive-error (funcall eval '()) "~A: 0個の入力" name)
      (signals nb:primitive-error
          (funcall eval (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)))
        "~A: 2個の入力" name))))

(test primitives/unary/non-float-dtype-signals-primitive-error
  ":i1 は浮動小数点ではないので PRIMITIVE-ERROR になる（neg/exp/log/tanh/sqrt/rsqrt/log1p/expm1）。"
  (dolist (name '(:neg :exp :log :tanh :sqrt :rsqrt :log1p :expm1))
    (signals nb:primitive-error
      (funcall (nb::primitive-abstract-eval (nb::find-primitive name))
               (list (nb:make-aval '(2 3) :i1)))
      "~A" name)))

(test primitives/minmax/wrong-arity-signals-primitive-error
  "max/min はちょうど2つの入力を要求する。0個・1個・3個ではどれも
PRIMITIVE-ERROR になる。"
  (dolist (name '(:max :min))
    (let ((eval (nb::primitive-abstract-eval (nb::find-primitive name))))
      (signals nb:primitive-error (funcall eval '()) "~A: 0個の入力" name)
      (signals nb:primitive-error (funcall eval (list (nb:make-aval '(2 3) :f32))) "~A: 1個の入力" name)
      (signals nb:primitive-error
          (funcall eval (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)))
        "~A: 3個の入力" name))))

(test primitives/minmax/shape-mismatch-signals-primitive-error
  "shape がずれた2つの入力を渡すと max/min のどちらも PRIMITIVE-ERROR になる。"
  (is (check-it (generator (uniform-integer :lo 1 :hi 4))
                (lambda (extra)
                  (every (lambda (name)
                           (let ((eval (nb::primitive-abstract-eval (nb::find-primitive name))))
                             (handler-case
                                 (progn
                                   (funcall eval
                                            (list (nb:make-aval '(2 3) :f32)
                                                  (nb:make-aval (list (+ 2 extra) 3) :f32)))
                                   nil)
                               (nb:primitive-error () t))))
                         '(:max :min)))
                :regression-id primitives/minmax/shape-mismatch-signals-primitive-error
                :regression-file (regression-path "primitives-minmax-shape-mismatch"))))

(test primitives/minmax/dtype-mismatch-signals-primitive-error
  "dtype の違う2つの入力を渡すと PRIMITIVE-ERROR になる。"
  (dolist (name '(:max :min))
    (signals nb:primitive-error
      (funcall (nb::primitive-abstract-eval (nb::find-primitive name))
               (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f64)))
      "~A" name)))

(test primitives/minmax/non-float-dtype-signals-primitive-error
  ":i1 は浮動小数点ではないので、shape/dtype が一致していても PRIMITIVE-ERROR
になる（max/min）。"
  (dolist (name '(:max :min))
    (signals nb:primitive-error
      (funcall (nb::primitive-abstract-eval (nb::find-primitive name))
               (list (nb:make-aval '(2 3) :i1) (nb:make-aval '(2 3) :i1)))
      "~A" name)))

;;; --- 性質4: golden emit テスト ---

(defmacro def-unary-golden-emit-test (test-name prim-name f32-fixture bf16-fixture)
  `(test ,test-name
     ,(format nil "~(~A~) の emit は tests/fixtures/stablehlo/ops/~A.mlir /
~A.mlir の op 行と、SSA 名を正規化したうえで一致する。" prim-name f32-fixture bf16-fixture)
     (let ((prim (nb::find-primitive ,prim-name)))
       (flet ((%check (fixture dtype)
                (let* ((aval (nb:make-aval '(4) dtype))
                       (emitted (funcall (nb::primitive-emit prim) '("%a") (list aval) "%0" aval))
                       (expected (first (%unary-fixture-op-lines fixture))))
                  (is (string= (%normalize-unary-ssa-names emitted) (%normalize-unary-ssa-names expected))
                      "~A: got ~S, expected ~S" fixture emitted expected))))
         (%check ,f32-fixture :f32)
         (%check ,bf16-fixture :bf16)))))

(def-unary-golden-emit-test primitives/neg/emit-matches-fixture :neg "negate" "negate_bf16")
(def-unary-golden-emit-test primitives/exp/emit-matches-fixture :exp "exponential" "exponential_bf16")
(def-unary-golden-emit-test primitives/log/emit-matches-fixture :log "log" "log_bf16")
(def-unary-golden-emit-test primitives/tanh/emit-matches-fixture :tanh "tanh" "tanh_bf16")
(def-unary-golden-emit-test primitives/sqrt/emit-matches-fixture :sqrt "sqrt" "sqrt_bf16")
(def-unary-golden-emit-test primitives/rsqrt/emit-matches-fixture :rsqrt "rsqrt" "rsqrt_bf16")
(def-unary-golden-emit-test primitives/log1p/emit-matches-fixture :log1p "log_plus_one" "log_plus_one_bf16")
(def-unary-golden-emit-test primitives/expm1/emit-matches-fixture :expm1 "exponential_minus_one" "exponential_minus_one_bf16")

(defmacro def-minmax-golden-emit-test (test-name prim-name f32-fixture bf16-fixture)
  `(test ,test-name
     ,(format nil "~(~A~) の emit は tests/fixtures/stablehlo/ops/~A.mlir /
~A.mlir の op 行と、SSA 名を正規化したうえで一致する。" prim-name f32-fixture bf16-fixture)
     (let ((prim (nb::find-primitive ,prim-name)))
       (flet ((%check (fixture dtype)
                (let* ((aval (nb:make-aval '(4 8) dtype))
                       (emitted (funcall (nb::primitive-emit prim) '("%a" "%b") (list aval aval) "%0" aval))
                       (expected (first (%unary-fixture-op-lines fixture))))
                  (is (string= (%normalize-unary-ssa-names emitted) (%normalize-unary-ssa-names expected))
                      "~A: got ~S, expected ~S" fixture emitted expected))))
         (%check ,f32-fixture :f32)
         (%check ,bf16-fixture :bf16)))))

(def-minmax-golden-emit-test primitives/max/emit-matches-fixture :max "maximum" "maximum_bf16")
(def-minmax-golden-emit-test primitives/min/emit-matches-fixture :min "minimum" "minimum_bf16")

;;; --- 追加のオラクル（reference-* と実装を共有しないため、mutation testing
;;; の生存対策になる） ---

(test primitives/neg/double-negation-is-identity
  "neg(neg(a)) は a と一致する。f32/f64 は正確に、bf16/f16 は raw storage の
ビット列がそのまま一致する（符号反転は丸め誤差を生まないため）。"
  (is (check-it (generator (array-spec :dtypes *dtypes*))
                (lambda (spec)
                  (let* ((dtype (array-spec-dtype spec))
                         (a (make-random-array spec))
                         (in-avals (list (nb:array-aval a dtype)))
                         (eager (nb::primitive-eager (nb::find-primitive :neg)))
                         (once (funcall eager (list a) in-avals))
                         (twice (funcall eager (list once) in-avals)))
                    (equalp twice a)))
                :regression-id primitives/neg/double-negation-is-identity
                :regression-file (regression-path "primitives-unary-neg-double-negation"))))

(test primitives/exp-log/round-trip-approximates-identity
  "正の a について exp(log(a)) は a に近い（許容誤差つき。bf16/f16 は
log→exp の2回丸めが入るため許容誤差を緩める）。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (a (make-random-array spec :seed seed :domain :positive))
                           (in-avals (list (nb:array-aval a dtype)))
                           (log-eager (nb::primitive-eager (nb::find-primitive :log)))
                           (exp-eager (nb::primitive-eager (nb::find-primitive :exp)))
                           (logged (funcall log-eager (list a) in-avals))
                           (result (funcall exp-eager (list logged) in-avals))
                           (rtol (if (member dtype '(:bf16 :f16)) 5d-2 nil))
                           (atol (if (member dtype '(:bf16 :f16)) 5d-3 nil)))
                      (multiple-value-bind (default-rtol default-atol) (dtype-tolerance dtype)
                        (allclose (%decode-unary-array result dtype) (%decode-unary-array a dtype)
                                  :rtol (or rtol default-rtol) :atol (or atol default-atol))))))
                :regression-id primitives/exp-log/round-trip-approximates-identity
                :regression-file (regression-path "primitives-unary-exp-log-round-trip"))))

(test primitives/max/result-is-at-least-both-operands
  "max(a, b) は a・b のどちらより小さくもなく（許容誤差つき）、a か b の
どちらかと一致する。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (a (make-random-array spec :seed seed))
                           (b (make-random-array spec :seed (1+ seed)))
                           (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype)))
                           (result (funcall (nb::primitive-eager (nb::find-primitive :max)) (list a b) in-avals))
                           (da (%decode-unary-array a dtype))
                           (db (%decode-unary-array b dtype))
                           (dr (%decode-unary-array result dtype)))
                      (dotimes (i (array-total-size dr) t)
                        (let ((r (row-major-aref dr i)) (x (row-major-aref da i)) (y (row-major-aref db i)))
                          (unless (and (>= r x) (>= r y) (or (= r x) (= r y)))
                            (return nil)))))))
                :regression-id primitives/max/result-is-at-least-both-operands
                :regression-file (regression-path "primitives-minmax-max-ge-both"))))

(test primitives/min/is-negated-max-of-negations
  "min(a, b) は -max(-a, -b) と一致する（許容誤差つき）。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (a (make-random-array spec :seed seed))
                           (b (make-random-array spec :seed (1+ seed)))
                           (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype)))
                           (neg-eager (nb::primitive-eager (nb::find-primitive :neg)))
                           (max-eager (nb::primitive-eager (nb::find-primitive :max)))
                           (min-eager (nb::primitive-eager (nb::find-primitive :min)))
                           (neg-a (funcall neg-eager (list a) (list (nb:array-aval a dtype))))
                           (neg-b (funcall neg-eager (list b) (list (nb:array-aval b dtype))))
                           (max-of-negations (funcall max-eager (list neg-a neg-b) in-avals))
                           (min-negated (funcall neg-eager (list max-of-negations) (list (nb:array-aval max-of-negations dtype))))
                           (min-result (funcall min-eager (list a b) in-avals)))
                      (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                        (allclose (%decode-unary-array min-result dtype) (%decode-unary-array min-negated dtype)
                                  :rtol rtol :atol atol)))))
                :regression-id primitives/min/is-negated-max-of-negations
                :regression-file (regression-path "primitives-minmax-min-is-negated-max"))))

;;; --- NaN・境界値の例ベーステスト（全 dtype） ---

(test primitives/unary/nan-propagates
  "NaN を入力すると neg/exp/log/tanh/sqrt/rsqrt/log1p/expm1 はすべて NaN を返す（全 dtype）。"
  (dolist (dtype *dtypes*)
    (let ((aval (nb:make-aval '() dtype))
          (nan-array (%unary-nan-array dtype)))
      (dolist (name '(:neg :exp :log :tanh :sqrt :rsqrt :log1p :expm1))
        (let ((result (funcall (nb::primitive-eager (nb::find-primitive name)) (list nan-array) (list aval))))
          (is (sb-ext:float-nan-p (%decode-unary-scalar dtype (row-major-aref result 0)))
              "~A/~A: NaN 入力の結果が NaN でない" name dtype))))))

(test primitives/minmax/nan-on-either-side-propagates
  "max/min は、どちらの引数が NaN でも NaN を返す（both orders。CL の
MAX/MIN が引数の順序で挙動を変える問題を再発させないための性質。全 dtype）。"
  (dolist (dtype *dtypes*)
    (let* ((aval (nb:make-aval '() dtype))
           (one (%unary-scalar-array dtype 1.0d0))
           (nan (%unary-nan-array dtype)))
      (dolist (name '(:max :min))
        (let ((eager (nb::primitive-eager (nb::find-primitive name))))
          (is (sb-ext:float-nan-p
               (%decode-unary-scalar dtype (row-major-aref (funcall eager (list nan one) (list aval aval)) 0)))
              "~A/~A: NaN, 1.0" name dtype)
          (is (sb-ext:float-nan-p
               (%decode-unary-scalar dtype (row-major-aref (funcall eager (list one nan) (list aval aval)) 0)))
              "~A/~A: 1.0, NaN" name dtype))))))

(defun %unary-eager-scalar (name dtype value)
  "NAME の eager 実装を VALUE（double）のスカラーに適用し、結果を double で返す。"
  (let ((aval (nb:make-aval '() dtype)))
    (%decode-unary-scalar
     dtype (row-major-aref (funcall (nb::primitive-eager (nb::find-primitive name))
                                    (list (%unary-scalar-array dtype value)) (list aval))
                           0))))

(test primitives/sqrt-rsqrt/negative-is-nan
  "負の入力は sqrt / rsqrt とも NaN になる（複素数を返さない。全 dtype、issue #196）。"
  (dolist (dtype *dtypes*)
    (dolist (name '(:sqrt :rsqrt))
      (is (sb-ext:float-nan-p (%unary-eager-scalar name dtype -4.0d0)) "~A/~A: -4" name dtype)
      (is (sb-ext:float-nan-p (%unary-eager-scalar name dtype -1.0d-3)) "~A/~A: -1e-3" name dtype))))

(test primitives/sqrt-rsqrt/exact-values-and-boundaries
  "sqrt(4)=2, rsqrt(4)=0.5, sqrt(0)=0, rsqrt(0)=+無限大, sqrt(-0)=-0 (全 dtype)。"
  (dolist (dtype *dtypes*)
    (is (= 2d0 (%unary-eager-scalar :sqrt dtype 4d0)) "sqrt ~A" dtype)
    (is (= 0.5d0 (%unary-eager-scalar :rsqrt dtype 4d0)) "rsqrt ~A" dtype)
    (is (= 0d0 (%unary-eager-scalar :sqrt dtype 0d0)) "sqrt 0 ~A" dtype)
    (let ((r (%unary-eager-scalar :rsqrt dtype 0d0)))
      (is (and (not (sb-ext:float-nan-p r)) (plusp r) (> r 1d30)) "rsqrt 0 ~A: ~S" dtype r))
    (let ((r (%unary-eager-scalar :rsqrt dtype -0d0)))
      (is (and (not (sb-ext:float-nan-p r)) (minusp r) (< r -1d30)) "rsqrt -0 ~A: ~S" dtype r))
    (let ((r (%unary-eager-scalar :sqrt dtype -0d0)))
      (is (and (zerop r) (minusp (float-sign r))) "sqrt -0 ~A: ~S" dtype r))))

(test primitives/sqrt-rsqrt/eager-matches-jax-fixture
  "sqrt / rsqrt の eager 実装は、JAX（jax.lax.sqrt / rsqrt。tests/fixtures/sqrt/generate.py）
のフィクスチャと全 dtype で一致する（NaN・±無限大・±0 は厳密、それ以外は dtype の許容誤差）。
入力には ±0・±inf・NaN・負の値を含む。f32 / f64 / bf16 の非正規数は XLA CPU が 0 に
丸めるので入れていない（generate.py）。"
  (dolist (dtype *dtypes*)
    (let* ((rows (load-sqrt-jax-fixture dtype))
           (x (sqrt-fixture-input-array dtype rows))
           (aval (nb:make-aval (list (length rows)) dtype)))
      (is (plusp (length rows)))
      (loop for (name index) in '((:sqrt 1) (:rsqrt 2))
            for result = (funcall (nb::primitive-eager (nb::find-primitive name)) (list x) (list aval))
            do (loop for row in rows for i from 0
                     do (is (sqrt-fixture-match-p dtype (aref result i) (nth index row))
                            "~A/~A: 入力ビット ~D, 期待 ~D, 実際 ~S"
                            name dtype (first row) (nth index row) (aref result i)))))))

(test primitives/rsqrt/public-function-on-arrays-and-numbers
  "nb:rsqrt は with-tracing の外でも呼べる: 配列には eager のプリミティブ、数値には
(/ 1 (sqrt x))。"
  (is (fboundp 'nb:rsqrt))
  (let ((result (nb:rsqrt (make-array 3 :element-type 'double-float :initial-contents '(4d0 1d0 0.25d0)))))
    (is (allclose result (make-array 3 :element-type 'double-float :initial-contents '(0.5d0 1d0 2d0)) :dtype :f64)))
  (is (= 0.5 (nb:rsqrt 4.0))))

(test primitives/sqrt-rsqrt/number-literals-in-with-tracing
  "with-tracing の中の数値リテラルの (sqrt 4) / (nb:rsqrt 4) は CL の演算で畳まれ、
トレーサと組める。"
  (let ((add-rsqrt (nb:trace-to-graph (nb:with-tracing (x) (+ x (nb:rsqrt 4))) (list (nb:make-aval '(2) :f64))))
        (add-sqrt (nb:trace-to-graph (nb:with-tracing (x) (+ x (sqrt 4))) (list (nb:make-aval '(2) :f64))))
        (x (make-array 2 :element-type 'double-float :initial-contents '(1d0 2d0))))
    (is (allclose (nb:eval-graph add-rsqrt x)
                  (make-array 2 :element-type 'double-float :initial-contents '(1.5d0 2.5d0)) :dtype :f64))
    (is (allclose (nb:eval-graph add-sqrt x)
                  (make-array 2 :element-type 'double-float :initial-contents '(3d0 4d0)) :dtype :f64))))

(test primitives/log1p-expm1/keep-the-digits-of-small-inputs
  "|x| が小さいとき log1p(x) と expm1(x) は x そのもの（相対誤差は dtype の丸め1回ぶん）。
(log (+ 1 x)) / (- (exp x) 1) の素朴な式だと 1 + x の丸めで x が消える（f32 の 1e-10 で 0）。
この差が softmax の Softmax Collapse の原因なので（docs/numerics.md）、桁が残ることを守る。"
  (dolist (dtype *dtypes*)
    ;; x^2/2 が dtype の丸めより小さい値（f64 は 1e-17 未満）。f16 では 0 に丸まる
    (dolist (x (if (eq dtype :f64) '(3d-20 -3d-20 1d-25) '(1d-10 -1d-10 3d-20 -3d-20)))
      (dolist (name '(:log1p :expm1))
        (let* ((rounded (%unary-eager-scalar :neg dtype (- x)))
               (got (%unary-eager-scalar name dtype x)))
          (is (<= (abs (- got rounded)) (* (abs rounded) (ecase dtype (:f64 1d-15) (:f32 1d-6) ((:bf16 :f16) 1d-2))))
              "~A/~A: x = ~S（dtype に丸めると ~S）, 結果 ~S" name dtype x rounded got))))))

(test primitives/log1p-expm1/boundaries
  "log1p(-1) = 負の無限大、log1p(x < -1) = NaN、log1p(+inf) = +inf、expm1(-inf) = -1、expm1(+inf) =
+inf、expm1 の大きな入力は +inf（溢れる）、どちらも ±0 は符号を保つ（全 dtype）。"
  (dolist (dtype *dtypes*)
    (let ((inf sb-ext:double-float-positive-infinity))
      (is (= (- inf) (%unary-eager-scalar :log1p dtype -1d0)) "log1p -1 ~A" dtype)
      (is (sb-ext:float-nan-p (%unary-eager-scalar :log1p dtype -2d0)) "log1p -2 ~A" dtype)
      (is (= inf (%unary-eager-scalar :log1p dtype inf)) "log1p +inf ~A" dtype)
      (is (= -1d0 (%unary-eager-scalar :expm1 dtype (- inf))) "expm1 -inf ~A" dtype)
      (is (= inf (%unary-eager-scalar :expm1 dtype inf)) "expm1 +inf ~A" dtype)
      (is (= inf (%unary-eager-scalar :expm1 dtype 1d4)) "expm1 1e4 ~A" dtype)
      (dolist (name '(:log1p :expm1))
        (let ((r (%unary-eager-scalar name dtype -0d0)))
          (is (and (zerop r) (minusp (float-sign r))) "~A -0 ~A: ~S" name dtype r))))))

(test primitives/log1p-expm1/public-functions-on-arrays-and-numbers
  "nb:log1p / nb:expm1 は with-tracing の外でも呼べる: 配列には eager のプリミティブ、数値には
その float の型で計算した値を返す（single-float は single-float のまま）。"
  (let ((x (make-array 2 :element-type 'double-float :initial-contents '(1d-30 1d0))))
    (is (allclose (nb:log1p x) (make-array 2 :element-type 'double-float :initial-contents (list 1d-30 (log 2d0))) :dtype :f64))
    (is (allclose (nb:expm1 x) (make-array 2 :element-type 'double-float :initial-contents (list 1d-30 (- (exp 1d0) 1))) :dtype :f64)))
  (is (typep (nb:log1p 1e-20) 'single-float))
  (is (= 1e-20 (nb:log1p 1e-20)))
  (is (= 1d-30 (nb:expm1 1d-30))))

(test primitives/log/zero-is-negative-infinity
  "log(0) は符号によらず負の無限大になる（境界値、全 dtype）。"
  (dolist (dtype *dtypes*)
    (let* ((aval (nb:make-aval '() dtype))
           (zero (%unary-scalar-array dtype 0.0d0))
           (result (funcall (nb::primitive-eager (nb::find-primitive :log)) (list zero) (list aval)))
           (value (%decode-unary-scalar dtype (row-major-aref result 0))))
      (is (and (sb-ext:float-infinity-p value) (minusp value)) "~A: log(0)" dtype))))

(test primitives/log/negative-is-nan
  "log(負の値) は NaN になる（CL の (log -1.0) が複素数を返す問題への
回帰。全 dtype）。"
  (dolist (dtype *dtypes*)
    (let* ((aval (nb:make-aval '() dtype))
           (negative (%unary-scalar-array dtype -1.0d0))
           (result (funcall (nb::primitive-eager (nb::find-primitive :log)) (list negative) (list aval))))
      (is (sb-ext:float-nan-p (%decode-unary-scalar dtype (row-major-aref result 0))) "~A: log(-1)" dtype))))

(test primitives/log/negative-is-canonical-quiet-nan
  "f32 / f64 の log(負の値) は、どの負の値でも canonical quiet NaN
（符号ビット 0・quiet ビット 1・残りのペイロード 0。f32 は #x7FC00000、
f64 は #x7FF8000000000000）のビット列になる。NaN であることだけを見る
性質では、ペイロードの違う NaN や signaling NaN（#x7FF7FFFF...）を返す
変異体が生き残っていた（issue #70）。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1)
                                  (uniform-real :lo -1d6 :hi -1d-6)))
                (lambda (args)
                  (destructuring-bind (dtype-index x) args
                    (let* ((dtype (nth dtype-index '(:f32 :f64)))
                           (aval (nb:make-aval '() dtype))
                           (negative (%unary-scalar-array dtype x))
                           (result (funcall (nb::primitive-eager (nb::find-primitive :log))
                                            (list negative) (list aval)))
                           (value (row-major-aref result 0)))
                      (ecase dtype
                        (:f32 (= #x7FC00000 (ldb (byte 32 0) (sb-kernel:single-float-bits value))))
                        (:f64 (and (= #x7FF80000 (ldb (byte 32 0) (sb-kernel:double-float-high-bits value)))
                                   (= 0 (sb-kernel:double-float-low-bits value))))))))
                :regression-id primitives/log/negative-is-canonical-quiet-nan
                :regression-file (regression-path "primitives-log-negative-is-canonical-quiet-nan"))
      "log(負の値) のビット列が canonical quiet NaN と一致しなかった"))

(test primitives/exp/overflow-is-positive-infinity
  "exp(大きい正の値) は正の無限大になる（オーバーフロー、全 dtype）。"
  (dolist (dtype *dtypes*)
    (let* ((aval (nb:make-aval '() dtype))
           ;; bf16/f16 は最大有限値が f32/f64 よりずっと小さいので、共通して
           ;; 確実にオーバーフローする 1.0d4 を使う。
           (large (%unary-scalar-array dtype 1.0d4))
           (result (funcall (nb::primitive-eager (nb::find-primitive :exp)) (list large) (list aval)))
           (value (%decode-unary-scalar dtype (row-major-aref result 0))))
      (is (and (sb-ext:float-infinity-p value) (plusp value)) "~A: exp(1e4)" dtype))))

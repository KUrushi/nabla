;;;; add / sub / mul / div の性質（issue #31 p1）。
;;;;
;;;; find-primitive / primitive-abstract-eval / primitive-emit /
;;;; primitive-eager は内部シンボル（nb::）で呼ぶ（プリミティブ名は
;;;; キーワードで、契約 §0 のとおり wave 2 は nabla から何も export しない）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; ゴールデン emit テストのための、この unit 専用のフィクスチャ読み込み
;;; ヘルパー（tests/iree/support.lisp の STABLEHLO-FIXTURE と同種だが、
;;; nabla/tests は nabla/iree/tests に依存しないため、ここに自分のコピーを
;;; 持つ。契約 §4 の「各 unit が自分の %fixture-op-lines / %normalize-ssa-names
;;; を持つ」方針どおり）。

(defun %read-op-fixture (name)
  "tests/fixtures/stablehlo/ops/NAME.mlir の内容を文字列で返す。"
  (let ((path (asdf:system-relative-pathname
               "nabla" (format nil "tests/fixtures/stablehlo/ops/~A.mlir" name))))
    (with-open-file (stream path :direction :input)
      (let ((text (make-string (file-length stream))))
        (subseq text 0 (read-sequence text stream))))))

(defun %split-lines (text)
  (with-input-from-string (s text)
    (loop for line = (read-line s nil nil)
          while line collect line)))

(defun %fixture-op-lines (name)
  "NAME フィクスチャの、func.func の行の次から func.return の行の前までの
行（前後の空白を trim したもの）のリストを返す。"
  (let* ((lines (%split-lines (%read-op-fixture name)))
         (start (position-if (lambda (l) (search "func.func" l)) lines))
         (end (position-if (lambda (l) (search "func.return" l)) lines)))
    (mapcar (lambda (l) (string-trim '(#\Space #\Tab) l))
            (subseq lines (1+ start) end))))

(defun %normalize-ssa-names (text)
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

;;; 0除算のテストで使う、rank 0 のスカラー配列。DTYPE-VALUE /
;;; ELEMENT-TYPE-FOR-DTYPE は nabla.tests.support から export していない
;;; 内部関数なので :: で参照する（tests/support/random-array.lisp）。

(defun %scalar-array (dtype value)
  (let ((array (make-array '() :element-type (nabla.tests.support::element-type-for-dtype dtype))))
    (setf (row-major-aref array 0) (nabla.tests.support::dtype-value dtype value))
    array))

(defun %float-plus-infinity-p (x)
  (and (sb-ext:float-infinity-p x) (plusp x)))

(defun %float-minus-infinity-p (x)
  (and (sb-ext:float-infinity-p x) (minusp x)))

(defun %decode-scalar (dtype value)
  "VALUE（DTYPE の格納表現を持つ1要素）を DOUBLE-FLOAT に戻す。

tests/support/random-array.lisp の DECODE-ELEMENT / DECODE-ARRAY ではなく
NB::DECODE-FLOAT16 を直接使う。DECODE-ELEMENT の f16/bf16 デコーダは無限大・
NaN のビットパターン（指数部が全1）を非正規化数と同じ計算式に落として誤った
デコードをする既知の問題があり（テスト用の乱数配列は有限の値しか作らない
ため、これまで表面化していない）、0除算や div のオーバーフローを扱う
このファイルのテストではその問題を避けるため NB::DECODE-FLOAT16
（src/float16.lisp、無限大・NaN を正しく扱う）を使う。"
  (if (member dtype '(:bf16 :f16))
      (coerce (nb::decode-float16 value dtype) 'double-float)
      (coerce value 'double-float)))

(defun %decode-arith-array (array dtype)
  "ARRAY（DTYPE の格納表現）を %DECODE-SCALAR で要素ごとにデコードした
DOUBLE-FLOAT の配列を返す。tests/support の DECODE-ARRAY の代わりに、この
ファイルのテストではすべてこちらを使う（理由は %DECODE-SCALAR 参照）。"
  (let ((result (make-array (array-dimensions array) :element-type 'double-float)))
    (dotimes (i (array-total-size array) result)
      (setf (row-major-aref result i) (%decode-scalar dtype (row-major-aref array i))))))

(defun %cast-double-array-to-dtype (array dtype)
  "DOUBLE-FLOAT の ARRAY を、いったん DTYPE の格納表現に丸めてから
DOUBLE-FLOAT に戻す。DTYPE の精度・表現範囲（bf16/f16 のオーバーフローで
無限大になることを含む）を、参照実装の期待値の側にも同じように反映させる
ために使う（div は正の値どうしでも、割る数が0に近ければ f16 の範囲
（最大約65504）を超えて無限大になりうる。これは実装の誤りではなく f16 の
表現力の限界なので、参照実装の DOUBLE-FLOAT の結果と直接比べると
オーバーフローの分だけ誤って不一致になってしまう）。"
  (let ((shape (array-dimensions array)))
    (ecase dtype
      (:f64 array)
      (:f32
       (let ((result (make-array shape :element-type 'double-float)))
         (dotimes (i (array-total-size array) result)
           (setf (row-major-aref result i)
                 (coerce (coerce (row-major-aref array i) 'single-float) 'double-float)))))
      ((:bf16 :f16)
       (let ((single (make-array shape :element-type 'single-float)))
         (dotimes (i (array-total-size array))
           (setf (row-major-aref single i) (coerce (row-major-aref array i) 'single-float)))
         (let ((bits (nb::encode-float16-array single dtype))
               (result (make-array shape :element-type 'double-float)))
           (dotimes (i (array-total-size single) result)
             (setf (row-major-aref result i)
                   (coerce (nb::decode-float16 (row-major-aref bits i) dtype) 'double-float)))))))))

(defun %arrays-close-with-inf (actual expected dtype)
  "ACTUAL と EXPECTED（同じ shape の DOUBLE-FLOAT 配列）の各要素が、無限大・
NaN を含めて一致するか判定する。ALLCLOSE（tests/support/allclose.lisp）は
無限大どうしの差分（inf - inf）を計算しようとして
FLOATING-POINT-INVALID-OPERATION を signal してしまうため、無限大・NaN は
先に符号／NaN かどうかで判定し、どちらも有限のときだけ DTYPE-TOLERANCE の
許容誤差で比べる（div のオーバーフローを含むこのファイル専用の比較）。"
  (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
    (dotimes (i (array-total-size actual) t)
      (let ((a (row-major-aref actual i))
            (b (row-major-aref expected i)))
        (unless (cond
                  ((or (sb-ext:float-nan-p a) (sb-ext:float-nan-p b))
                   (and (sb-ext:float-nan-p a) (sb-ext:float-nan-p b)))
                  ((or (sb-ext:float-infinity-p a) (sb-ext:float-infinity-p b))
                   (and (sb-ext:float-infinity-p a) (sb-ext:float-infinity-p b)
                        (= (float-sign a) (float-sign b))))
                  (t (approx= a b :rtol rtol :atol atol)))
          (return nil))))))

(defun %array-every (pred array)
  "ARRAY の全要素（row-major）が PRED を満たせば真を返す。多次元配列にも
使える（CL:EVERY は VECTOR にしか使えないため）。"
  (dotimes (i (array-total-size array) t)
    (unless (funcall pred (row-major-aref array i))
      (return nil))))

;;; --- 性質1: aval(eager) = abstract-eval ---

(defmacro def-aval-matches-abstract-eval-test (test-name prim-name)
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
                   :regression-file (regression-path ,(format nil "primitives-arith-~(~A~)-aval" prim-name)))
         ,(format nil "~A: eager の結果の aval が abstract-eval と一致しなかった" prim-name))))

(def-aval-matches-abstract-eval-test primitives/add/aval-matches-abstract-eval :add)
(def-aval-matches-abstract-eval-test primitives/sub/aval-matches-abstract-eval :sub)
(def-aval-matches-abstract-eval-test primitives/mul/aval-matches-abstract-eval :mul)
(def-aval-matches-abstract-eval-test primitives/div/aval-matches-abstract-eval :div)

;;; --- 性質2: eager = 参照実装（許容誤差つき） ---

(defmacro def-eager-matches-reference-test (test-name prim-name reference-fn domain)
  `(test ,test-name
     ,(format nil "~(~A~) の eager 実装の結果は、%decode-arith-array で double-float に
戻したうえで ~A と全 dtype・rtol/atol で一致する。" prim-name reference-fn)
     (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                      (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                   (lambda (args)
                     (destructuring-bind (spec seed) args
                       (let* ((dtype (array-spec-dtype spec))
                              (a (make-random-array spec :seed seed :domain ,domain))
                              (b (make-random-array spec :seed (1+ seed) :domain ,domain))
                              (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype)))
                              (result (funcall (nb::primitive-eager (nb::find-primitive ,prim-name))
                                                (list a b) in-avals)))
                         (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                           (allclose (%decode-arith-array result dtype)
                                     (,reference-fn (%decode-arith-array a dtype) (%decode-arith-array b dtype))
                                     :rtol rtol :atol atol)))))
                   :regression-id ,test-name
                   :regression-file (regression-path ,(format nil "primitives-arith-~(~A~)-reference" prim-name))))))

(def-eager-matches-reference-test primitives/add/eager-matches-reference :add reference-add :any)
(def-eager-matches-reference-test primitives/sub/eager-matches-reference :sub reference-sub :any)
(def-eager-matches-reference-test primitives/mul/eager-matches-reference :mul reference-mul :any)

;; div は add/sub/mul と2点違う: (1) 0除算を避けるため :positive で生成する
;; （0除算そのものの挙動は下の PRIMITIVES/DIV/DIVISION-BY-ZERO-DOES-NOT-SIGNAL
;; で別に確かめる）。(2) 割る数が0に近いと、正の値どうしでも商が f16 の
;; 表現範囲（最大約65504）を超えて無限大になりうる（実装の誤りではなく f16
;; の限界）。REFERENCE-DIV の DOUBLE-FLOAT の結果を直接比べるとその分だけ
;; 誤って不一致になるため、%CAST-DOUBLE-ARRAY-TO-DTYPE で REFERENCE-DIV の
;; 結果も同じ DTYPE の精度・範囲に丸めてから比べる。
(test primitives/div/eager-matches-reference
  "div の eager 実装の結果は、%decode-arith-array で double-float に戻した
うえで、REFERENCE-DIV の結果を同じ dtype に丸めたものと全 dtype・rtol/atol
で一致する。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (a (make-random-array spec :seed seed :domain :positive))
                           (b (make-random-array spec :seed (1+ seed) :domain :positive))
                           (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype)))
                           (result (funcall (nb::primitive-eager (nb::find-primitive :div)) (list a b) in-avals)))
                      (%arrays-close-with-inf
                       (%decode-arith-array result dtype)
                       (%cast-double-array-to-dtype
                        (reference-div (%decode-arith-array a dtype) (%decode-arith-array b dtype))
                        dtype)
                       dtype))))
                :regression-id primitives/div/eager-matches-reference
                :regression-file (regression-path "primitives-arith-div-reference"))))

;;; --- 性質3: 不正な入力は PRIMITIVE-ERROR ---

(test primitives/arith/wrong-arity-signals-primitive-error
  "add/sub/mul/div はちょうど2つの入力を要求する。0個・1個・3個ではどれも
PRIMITIVE-ERROR になる。"
  (dolist (name '(:add :sub :mul :div))
    (let ((eval (nb::primitive-abstract-eval (nb::find-primitive name))))
      (signals nb:primitive-error (funcall eval '()) "~A: 0個の入力" name)
      (signals nb:primitive-error (funcall eval (list (nb:make-aval '(2 3) :f32))) "~A: 1個の入力" name)
      (signals nb:primitive-error
          (funcall eval (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)))
        "~A: 3個の入力" name))))

(test primitives/arith/shape-mismatch-signals-primitive-error
  "shape がずれた2つの入力を渡すと、add/sub/mul/div のすべてで PRIMITIVE-ERROR
になる（生成器で shape をずらす。tests/primitive-test.lisp と同じ手法）。"
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
                         '(:add :sub :mul :div)))
                :regression-id primitives/arith/shape-mismatch-signals-primitive-error
                :regression-file (regression-path "primitives-arith-shape-mismatch"))))

(test primitives/arith/dtype-mismatch-signals-primitive-error
  "dtype の違う2つの入力を渡すと PRIMITIVE-ERROR になる。"
  (dolist (name '(:add :sub :mul :div))
    (signals nb:primitive-error
      (funcall (nb::primitive-abstract-eval (nb::find-primitive name))
               (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f64)))
      "~A" name)))

(test primitives/arith/non-float-dtype-signals-primitive-error
  ":i1 は浮動小数点ではないので、shape/dtype が一致していても PRIMITIVE-ERROR
になる。"
  (dolist (name '(:add :sub :mul :div))
    (signals nb:primitive-error
      (funcall (nb::primitive-abstract-eval (nb::find-primitive name))
               (list (nb:make-aval '(2 3) :i1) (nb:make-aval '(2 3) :i1)))
      "~A" name)))

;;; --- 性質4: golden emit テスト ---

(defmacro def-golden-emit-test (test-name prim-name f32-fixture bf16-fixture)
  `(test ,test-name
     ,(format nil "~(~A~) の emit は tests/fixtures/stablehlo/ops/~A.mlir /
~A.mlir の op 行と、SSA 名を正規化したうえで一致する。" prim-name f32-fixture bf16-fixture)
     (let ((prim (nb::find-primitive ,prim-name)))
       (flet ((%check (fixture dtype)
                (let* ((aval (nb:make-aval '(4 8) dtype))
                       (emitted (funcall (nb::primitive-emit prim) '("%a" "%b") (list aval aval) "%0" aval))
                       (expected (first (%fixture-op-lines fixture))))
                  (is (string= (%normalize-ssa-names emitted) (%normalize-ssa-names expected))
                      "~A: got ~S, expected ~S" fixture emitted expected))))
         (%check ,f32-fixture :f32)
         (%check ,bf16-fixture :bf16)))))

(def-golden-emit-test primitives/add/emit-matches-fixture :add "add" "add_bf16")
(def-golden-emit-test primitives/sub/emit-matches-fixture :sub "subtract" "subtract_bf16")
(def-golden-emit-test primitives/mul/emit-matches-fixture :mul "multiply" "multiply_bf16")
(def-golden-emit-test primitives/div/emit-matches-fixture :div "divide" "divide_bf16")

;;; --- 追加のオラクル（reference-* と実装を共有しないため、係数の入れ替え
;;; のような変異を独立に検出できる。mutation testing の生存対策） ---

(test primitives/add/is-commutative
  "add(a, b) と add(b, a) は（浮動小数点の丸め誤差の範囲で）一致する。"
  (is (check-it (generator (tuple (array-spec :dtypes *dtypes*)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (a (make-random-array spec :seed seed))
                           (b (make-random-array spec :seed (1+ seed)))
                           (in-avals (list (nb:array-aval a dtype) (nb:array-aval b dtype)))
                           (eager (nb::primitive-eager (nb::find-primitive :add))))
                      (allclose (%decode-arith-array (funcall eager (list a b) in-avals) dtype)
                                (%decode-arith-array (funcall eager (list b a) in-avals) dtype)))))
                :regression-id primitives/add/is-commutative
                :regression-file (regression-path "primitives-arith-add-commutative"))))

(test primitives/sub/self-is-zero
  "sub(a, a) はどの要素も0になる（浮動小数点の丸め誤差なしで正確に0）。"
  (is (check-it (generator (array-spec :dtypes *dtypes*))
                (lambda (spec)
                  (let* ((dtype (array-spec-dtype spec))
                         (a (make-random-array spec))
                         (in-avals (list (nb:array-aval a dtype) (nb:array-aval a dtype)))
                         (result (funcall (nb::primitive-eager (nb::find-primitive :sub)) (list a a) in-avals)))
                    (%array-every #'zerop (%decode-arith-array result dtype))))
                :regression-id primitives/sub/self-is-zero
                :regression-file (regression-path "primitives-arith-sub-self-is-zero"))))

(test primitives/mul/by-ones-is-identity
  "mul(a, ones) は a と（許容誤差つきで）一致する。"
  (is (check-it (generator (array-spec :dtypes *dtypes*))
                (lambda (spec)
                  (let* ((dtype (array-spec-dtype spec))
                         (a (make-random-array spec))
                         (ones (make-array (array-spec-shape spec)
                                            :element-type (array-element-type a)
                                            :initial-element (nabla.tests.support::dtype-value dtype 1.0d0)))
                         (in-avals (list (nb:array-aval a dtype) (nb:array-aval ones dtype)))
                         (result (funcall (nb::primitive-eager (nb::find-primitive :mul)) (list a ones) in-avals)))
                    (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                      (allclose (%decode-arith-array result dtype) (%decode-arith-array a dtype) :rtol rtol :atol atol))))
                :regression-id primitives/mul/by-ones-is-identity
                :regression-file (regression-path "primitives-arith-mul-by-ones"))))

(test primitives/div/self-is-one-for-positive
  "正の a について div(a, a) は1になる（許容誤差つき）。"
  (is (check-it (generator (array-spec :dtypes *dtypes*))
                (lambda (spec)
                  (let* ((dtype (array-spec-dtype spec))
                         (a (make-random-array spec :domain :positive))
                         (in-avals (list (nb:array-aval a dtype) (nb:array-aval a dtype)))
                         (result (funcall (nb::primitive-eager (nb::find-primitive :div)) (list a a) in-avals)))
                    (%array-every (lambda (x) (approx= x 1.0d0 :dtype dtype))
                                  (%decode-arith-array result dtype))))
                :regression-id primitives/div/self-is-one-for-positive
                :regression-file (regression-path "primitives-arith-div-self-is-one"))))

;;; --- 0除算: signal せず IEEE 754 の ±inf / NaN を返す ---

(test primitives/div/division-by-zero-does-not-signal
  "1/0 → +inf、-1/0 → -inf、0/0 → NaN になり、どのdtypeでも
DIVISION-BY-ZERO 等の浮動小数点コンディションを signal しない。"
  (dolist (dtype *dtypes*)
    (let ((eager (nb::primitive-eager (nb::find-primitive :div)))
          (aval (nb:make-aval '() dtype)))
      (flet ((%run (a b)
               (%decode-scalar dtype
                                (row-major-aref
                                 (funcall eager (list (%scalar-array dtype a) (%scalar-array dtype b))
                                          (list aval aval))
                                 0))))
        (is (%float-plus-infinity-p (finishes (%run 1.0d0 0.0d0))) "~A: 1/0" dtype)
        (is (%float-minus-infinity-p (finishes (%run -1.0d0 0.0d0))) "~A: -1/0" dtype)
        (is (sb-ext:float-nan-p (finishes (%run 0.0d0 0.0d0))) "~A: 0/0" dtype)))))

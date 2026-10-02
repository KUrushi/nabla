;;;; nb:defprimitive / find-primitive / tensor-type-string の性質（issue #29、u1a）。
;;;;
;;;; find-primitive / make-eqn は内部シンボル（nb::）で呼ぶ。テストは内部を
;;;; nb:: で使っている。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(test primitive/unknown-name-signals-unknown-primitive
  "登録されていない名前を make-eqn に渡すと UNKNOWN-PRIMITIVE が signal され、
UNKNOWN-PRIMITIVE-NAME が渡した名前をそのまま返す。"
  (let ((v (nb::make-var (nb:make-aval '(2 3) :f32)))
        (entered-handler nil))
    (handler-case
        (nb::make-eqn :%test-does-not-exist (list v))
      (nb:unknown-primitive (condition)
        (setf entered-handler t)
        (is (eq :%test-does-not-exist (nb:unknown-primitive-name condition)))))
    (is (eq t entered-handler) "make-eqn が UNKNOWN-PRIMITIVE を signal しなかった")))

(test primitive/bad-param-keys-signal-primitive-error
  "未知のキー、欠落したキー、余分なキーはすべて PRIMITIVE-ERROR になる。"
  (let ((v (nb::make-var (nb:make-aval '(2 3) :f32))))
    (signals nb:primitive-error (nb::make-eqn :%test-reshape (list v) :typo '(6)))
    (signals nb:primitive-error (nb::make-eqn :%test-reshape (list v)))
    (signals nb:primitive-error (nb::make-eqn :%test-reshape (list v) :shape '(6) :extra 1))))

(test primitive/test-add-shape-mismatch-signals-primitive-error
  "%test-add に shape の違う2つの入力を渡すと PRIMITIVE-ERROR になる（生成器で
shape をずらす）。"
  (is (check-it (generator (uniform-integer :lo 1 :hi 4))
                (lambda (extra)
                  (let ((a (nb::make-var (nb:make-aval '(2 3) :f32)))
                        (b (nb::make-var (nb:make-aval (list (+ 2 extra) 3) :f32))))
                    (handler-case
                        (progn (nb::make-eqn :%test-add (list a b)) nil)
                      (nb:primitive-error () t))))
                :regression-id primitive/test-add-shape-mismatch-signals-primitive-error
                :regression-file (regression-path "primitive-test-add-shape-mismatch"))))

(test primitive/test-add-dtype-mismatch-signals-primitive-error
  "%test-add に dtype の違う2つの入力を渡すと PRIMITIVE-ERROR になる。"
  (let ((a (nb::make-var (nb:make-aval '(2 3) :f32)))
        (b (nb::make-var (nb:make-aval '(2 3) :f64))))
    (signals nb:primitive-error (nb::make-eqn :%test-add (list a b)))))

(test primitive/defprimitive-redefinition-replaces-registration
  "DEFPRIMITIVE を再評価すると FIND-PRIMITIVE の返り値が EQ でなくなり、
振る舞いも新しい定義に変わる。"
  (nb:defprimitive %test-redefine-me ()
    :abstract-eval (lambda (in-avals) (declare (ignore in-avals)) (nb:make-aval '(1) :f32)))
  (let ((first (nb::find-primitive :%test-redefine-me)))
    (nb:defprimitive %test-redefine-me ()
      :abstract-eval (lambda (in-avals) (declare (ignore in-avals)) (nb:make-aval '(2) :f32)))
    (let ((second (nb::find-primitive :%test-redefine-me)))
      (is (not (eq first second)))
      (let ((eqn (nb::make-eqn :%test-redefine-me '())))
        (is (equal '(2) (nb:aval-shape (nb:var-aval (first (nb:eqn-outvars eqn))))))))))

(test primitive/defprimitive-requires-abstract-eval
  ":ABSTRACT-EVAL 無しの DEFPRIMITIVE はマクロ展開時にエラーになる。"
  (signals error (macroexpand-1 '(nb:defprimitive %test-no-abstract-eval ()))))

(test primitive/defprimitive-requires-keyword-params
  "PARAM-KEYWORDS に非キーワードを渡すとマクロ展開時にエラーになる。"
  (signals error
    (macroexpand-1 '(nb:defprimitive %test-bad-params (shape)
                      :abstract-eval (lambda (in-avals &key shape) (declare (ignore shape)) (first in-avals))))))

(test primitive/defprimitive-rejects-unknown-keys
  "ABSTRACT-EVAL / EMIT / EAGER / JVP / TRANSPOSE 以外のキー（例 :BATCH）を渡すとエラーになる。"
  (signals error
    (macroexpand-1 '(nb:defprimitive %test-unknown-key ()
                      :abstract-eval (lambda (in-avals) (first in-avals))
                      :batch (lambda () nil)))))

(test primitive/tensor-type-string-matches-shape-and-dtype
  "TENSOR-TYPE-STRING は \"tensor<\" + shape の各次元と dtype 名を x で
繋いだもの + \">\" に一致する。オラクルは実装とは別の組み立て方（shape と
dtype 名を1本のリストにしてから x で繋ぐ）で計算する。"
  (is (check-it (generator (array-spec))
                (lambda (spec)
                  (let ((aval (nb:make-aval (array-spec-shape spec) (array-spec-dtype spec))))
                    (string= (nb::tensor-type-string aval)
                             (format nil "tensor<~A>"
                                     (format nil "~{~A~^x~}"
                                             (append (array-spec-shape spec)
                                                     (list (nb::dtype-mlir-name (array-spec-dtype spec)))))))))
                :regression-id primitive/tensor-type-string-matches-shape-and-dtype
                :regression-file (regression-path "primitive-tensor-type-string"))))

(test primitive/tensor-type-string-fixed-examples
  "TENSOR-TYPE-STRING の既知の入出力の組（rank 0、要素数0の次元を含む多次元、
bf16）を固定値で確かめる。"
  (is (string= "tensor<f32>" (nb::tensor-type-string (nb:make-aval '() :f32))))
  (is (string= "tensor<2x3xf32>" (nb::tensor-type-string (nb:make-aval '(2 3) :f32))))
  (is (string= "tensor<0x3xbf16>" (nb::tensor-type-string (nb:make-aval '(0 3) :bf16)))))

(test primitive/dtype-mlir-name-covers-all-dtypes
  "DTYPE-MLIR-NAME は f32/f64/bf16/f16/i1 のすべてに対応する。"
  (is (string= "f32" (nb::dtype-mlir-name :f32)))
  (is (string= "f64" (nb::dtype-mlir-name :f64)))
  (is (string= "bf16" (nb::dtype-mlir-name :bf16)))
  (is (string= "f16" (nb::dtype-mlir-name :f16)))
  (is (string= "i1" (nb::dtype-mlir-name :i1))))

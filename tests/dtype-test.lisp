;;;; nb:dtype 回りの性質（issue #7）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(test dtype/array-dtype/round-trips-with-spec-dtype
  "array-dtype に、元の spec の dtype を明示的に渡すと、その dtype を
そのまま返す（f32 / f64 / bf16 / f16 のすべてで）。"
  (is (check-it (generator (array-spec))
                (lambda (spec)
                  (let ((array (make-random-array spec)))
                    (eq (nb:array-dtype array (array-spec-dtype spec))
                        (array-spec-dtype spec))))
                :regression-id dtype/array-dtype/round-trips-with-spec-dtype
                :regression-file (regression-path "dtype-array-dtype-round-trips"))))

(test dtype/array-dtype/infers-f32-and-f64-without-dtype-argument
  "f32 / f64 の配列は DTYPE を渡さなくても array-dtype で正しく推論できる
（(unsigned-byte 16) の配列だけが bf16 / f16 の間で曖昧なので DTYPE が
必要になる）。"
  (is (check-it (generator (array-spec :dtypes '(:f32 :f64)))
                (lambda (spec)
                  (let ((array (make-random-array spec)))
                    (eq (nb:array-dtype array) (array-spec-dtype spec))))
                :regression-id dtype/array-dtype/infers-f32-and-f64-without-dtype-argument
                :regression-file (regression-path "dtype-array-dtype-infers-f32-f64"))))

(test dtype/array-dtype/u16-without-dtype-signals-mismatch
  "(unsigned-byte 16) の配列（bf16 / f16 用）に DTYPE を渡さないと
DTYPE-MISMATCH が signal される。"
  (is (check-it (generator (array-spec :dtypes '(:bf16 :f16)))
                (lambda (spec)
                  (let ((array (make-random-array spec)))
                    (handler-case
                        (progn (nb:array-dtype array) nil)
                      (nb:dtype-mismatch () t))))
                :regression-id dtype/array-dtype/u16-without-dtype-signals-mismatch
                :regression-file (regression-path "dtype-array-dtype-u16-without-dtype"))))

(test dtype/array-dtype/wrong-dtype-argument-signals-mismatch
  "bf16 の配列に :f32 を DTYPE として渡すと（要素型と食い違うので）
DTYPE-MISMATCH が signal される。"
  (let ((array (make-random-array (make-array-spec '(2 3) :bf16))))
    (signals nb:dtype-mismatch (nb:array-dtype array :f32))))

(test dtype/array-dtype/unsupported-element-type-signals-mismatch
  "single-float / double-float / (unsigned-byte 16) のどれでもない配列は
DTYPE-MISMATCH になる。"
  (signals nb:dtype-mismatch (nb:array-dtype (make-array 3 :element-type 'fixnum))))

(test dtype/dtype-element-type/matches-make-random-array
  "dtype-element-type が返す型は、make-random-array がその dtype で作る
配列の実際の要素型を含む（subtypep で比較し、upgraded-array-element-type
がより広い型を選ぶ処理系差を許容する）。"
  (is (check-it (generator (array-spec))
                (lambda (spec)
                  (let ((array (make-random-array spec)))
                    (subtypep (array-element-type array)
                              (nb:dtype-element-type (array-spec-dtype spec)))))
                :regression-id dtype/dtype-element-type/matches-make-random-array
                :regression-file (regression-path "dtype-element-type-matches-make-random-array"))))

(test dtype/dtype-element-type/rejects-non-dtype
  "dtype-element-type は DTYPE 型でないキーワードを拒否する。"
  (signals error (nb:dtype-element-type :not-a-dtype)))

(test dtype/dtype-byte-width/matches-known-widths
  "dtype-byte-width は f32→4, f64→8, bf16/f16→2。"
  (is (= 4 (nb:dtype-byte-width :f32)))
  (is (= 8 (nb:dtype-byte-width :f64)))
  (is (= 2 (nb:dtype-byte-width :bf16)))
  (is (= 2 (nb:dtype-byte-width :f16))))

(test dtype/dtype-mismatch/readers-expose-initargs
  "DTYPE-MISMATCH の2つのリーダーが、signal 時に渡した initarg をそのまま
返す。handler-case の節が実際に実行されたことも確かめる（そうしないと、
array-dtype が dtype-mismatch を signal しなくなる回帰が起きても、この
テストは何も検査しないまま無言で通ってしまう）。"
  (let ((array (make-array 3 :element-type '(unsigned-byte 16)))
        (entered-handler nil))
    (handler-case
        (nb:array-dtype array)
      (nb:dtype-mismatch (condition)
        (setf entered-handler t)
        (is (equal (array-element-type array) (nb:dtype-mismatch-element-type condition)))
        (is (null (nb:dtype-mismatch-dtype condition)))))
    (is (eq t entered-handler) "array-dtype が DTYPE-MISMATCH を signal しなかった")))

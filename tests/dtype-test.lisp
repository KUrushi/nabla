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

(test dtype/array-dtype/strict-subtypes-of-supported-types-signal-mismatch
  "SINGLE-FLOAT / (UNSIGNED-BYTE 16) の真の部分型（SUBTYPEP ではなく要素型が
一致するかどうかで判定しなければならない）は DTYPE-MISMATCH になる。
(UNSIGNED-BYTE 8) は (UNSIGNED-BYTE 16) の部分型だが 16 ビットではないため
bf16 / f16 として受理してはならず、要素型 NIL の配列（空型。SUBTYPEP NIL
SINGLE-FLOAT は真）を f32 として受理してもならない。これらを SUBTYPEP で
分類すると、要素数と無関係にホストの読み取り幅を決める to-device が
ヒープを読み越えてしまう（メモリ安全性のバグ）。BIT はそれ自体が :i1 の
要素型なので（issue #37）、:f16 のような別の DTYPE を渡したときだけ
DTYPE-MISMATCH になる（DTYPE 無しなら :i1 と推論される。それは下の
i1-round-trips で確かめる）。"
  (signals nb:dtype-mismatch (nb:array-dtype (make-array 6 :element-type '(unsigned-byte 8)) :bf16))
  (signals nb:dtype-mismatch (nb:array-dtype (make-array 6 :element-type 'bit) :f16))
  (signals nb:dtype-mismatch (nb:array-dtype (make-array 6 :element-type nil))))

(test dtype/array-dtype/i1-round-trips
  "BIT 配列は array-dtype に :i1 を渡すとそのまま返り、DTYPE を渡さなくても
:i1 と推論できる（(unsigned-byte 16) と違い BIT は他の dtype と紛れない）。"
  (is (check-it (generator (array-spec :dtypes '(:i1)))
                (lambda (spec)
                  (let ((array (make-random-array spec)))
                    (and (eq (nb:array-dtype array (array-spec-dtype spec)) :i1)
                         (eq (nb:array-dtype array) :i1))))
                :regression-id dtype/array-dtype/i1-round-trips
                :regression-file (regression-path "dtype-array-dtype-i1-round-trips"))))

(test dtype/array-dtype/i1-wrong-dtype-argument-signals-mismatch
  "BIT 配列に :f32 を DTYPE として渡すと（要素型と食い違うので）
DTYPE-MISMATCH が signal される。"
  (let ((array (make-random-array (make-array-spec '(2 3) :i1))))
    (signals nb:dtype-mismatch (nb:array-dtype array :f32))))

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
  "dtype-byte-width は f32→4, f64→8, bf16/f16→2, i1→1。"
  (is (= 4 (nb:dtype-byte-width :f32)))
  (is (= 8 (nb:dtype-byte-width :f64)))
  (is (= 2 (nb:dtype-byte-width :bf16)))
  (is (= 2 (nb:dtype-byte-width :f16)))
  (is (= 1 (nb:dtype-byte-width :i1))))

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

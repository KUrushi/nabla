;;;; make-random-array: array-spec から決定的な乱数配列を作る。

(in-package #:nabla.tests.support)

(defun element-type-for-dtype (dtype)
  "DTYPE を持つ配列の Common Lisp 上の element-type を返す。

f32 は single-float、f64 は double-float、bf16 / f16 はビット列を
そのまま持つ (unsigned-byte 16) にする（CLAUDE.md の約束）。i1（issue #37）
は bit。"
  (ecase dtype
    (:f32 'single-float)
    (:f64 'double-float)
    ((:bf16 :f16) '(unsigned-byte 16))
    (:i1 'bit)))

(defun dtype-value (dtype value)
  "DOUBLE-FLOAT の VALUE を、DTYPE の配列要素として格納する値に変換する。

bf16 / f16 は src/float16.lisp の NABLA::ENCODE-FLOAT16（RNE、issue #38）
に変換を委ねる。ここで独自のビット変換を持たない（issue #55）。i1
（issue #37）は BIT なので、VALUE の符号で 0 / 1 に落とす。"
  (ecase dtype
    (:f64 (coerce value 'double-float))
    (:f32 (coerce value 'single-float))
    ((:bf16 :f16) (nabla::encode-float16 (coerce value 'single-float) dtype))
    (:i1 (if (minusp value) 0 1))))

(defun decode-element (dtype value)
  "配列に格納されている VALUE を、比較用の DOUBLE-FLOAT に戻す。

bf16 / f16 は src/float16.lisp の NABLA::DECODE-FLOAT16（issue #38）に
変換を委ねる。以前は独自のビット変換を持っていて、無限大・NaN・
非正規化数を誤ってデコードしていた（issue #55）。SINGLE-FLOAT の NaN を
DOUBLE-FLOAT に coerce する操作そのものが SBCL の既定の :invalid トラップ
を踏むので、他のプリミティブ（src/primitives/common.lisp）と同じく
WITH-FLOAT-TRAPS-MASKED で包む。i1（issue #37）の VALUE は BIT（0 / 1）
そのものなので、そのまま DOUBLE-FLOAT に coerce する。"
  (ecase dtype
    ((:f32 :f64) (coerce value 'double-float))
    ((:bf16 :f16)
     (sb-int:with-float-traps-masked (:invalid)
       (coerce (nabla::decode-float16 value dtype) 'double-float)))
    (:i1 (coerce value 'double-float))))

(defun %random-domain-value (state domain)
  "STATE から DOMAIN に応じた DOUBLE-FLOAT を1つ取り出す。

:any は概ね [-1, 1)、:positive は (0, 1]、:unit は [0, 1) になる。"
  (ecase domain
    (:any (- (* 2.0d0 (random 1.0d0 state)) 1.0d0))
    (:positive (- 1.0d0 (random 1.0d0 state)))
    (:unit (random 1.0d0 state))))

(defun %smallest-positive-dtype-value (dtype)
  "DTYPE の格納表現で表せる、最小の正の値を返す。

:positive ドメイン用に 0 をクランプする値として使う。bf16 / f16 は
ビット列そのものが値なので、最小の正の非正規化数のビット列 (1) を返す。"
  (ecase dtype
    (:f64 least-positive-double-float)
    (:f32 least-positive-single-float)
    ((:bf16 :f16) 1)))

(defun %zero-dtype-value-p (dtype value)
  "DTYPE の格納表現の VALUE が +0 または -0 かどうかを返す。"
  (ecase dtype
    ((:f32 :f64) (zerop value))
    ((:bf16 :f16) (zerop (logand value #x7FFF)))))

(defun decode-array (array dtype)
  "ARRAY（DTYPE の格納表現を持つ配列。bf16 / f16 ならビット列）と同じ shape の
DOUBLE-FLOAT 配列を返す。各要素は DECODE-ELEMENT でデコードする。

allclose の :DTYPE 引数は ACTUAL と EXPECTED の両方を同じ DTYPE でデコード
する前提なので、reference-*（常に DOUBLE-FLOAT を返す）の期待値と比べる
ときはこちらで先にデコードしてから allclose を dtype 無しで呼ぶ（issue #12）。"
  (let ((result (make-array (array-dimensions array) :element-type 'double-float)))
    (dotimes (i (array-total-size array) result)
      (setf (row-major-aref result i) (decode-element dtype (row-major-aref array i))))))

(defun make-random-array (spec &key (seed 0) (domain :any))
  "SPEC (array-spec) と SEED から決定的な配列を作る。

SPEC の shape がそのまま array-dimensions になり、dtype に応じた
element-type を持つ。DOMAIN は :any (既定) / :positive / :unit。

DOMAIN が :positive のとき、生成した DOUBLE-FLOAT が 0 に十分近いと、
bf16 / f16 のように仮数部が狭い dtype では格納表現へ変換する際に
アンダーフローしてちょうど 0 になり得る。これは (0, 1] という契約に
反するので、0 になった要素はその dtype で表現できる最小の正の値に
クランプする。"
  (let* ((shape (array-spec-shape spec))
         (dtype (array-spec-dtype spec))
         (state (sb-ext:seed-random-state seed))
         (array (make-array shape :element-type (element-type-for-dtype dtype))))
    (dotimes (i (array-total-size array) array)
      (let ((value (dtype-value dtype (%random-domain-value state domain))))
        (when (and (eq domain :positive) (%zero-dtype-value-p dtype value))
          (setf value (%smallest-positive-dtype-value dtype)))
        (setf (row-major-aref array i) value)))))

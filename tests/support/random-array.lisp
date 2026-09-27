;;;; make-random-array: array-spec から決定的な乱数配列を作る。

(in-package #:nabla.tests.support)

(defun element-type-for-dtype (dtype)
  "DTYPE を持つ配列の Common Lisp 上の element-type を返す。

f32 は single-float、f64 は double-float、bf16 / f16 はビット列を
そのまま持つ (unsigned-byte 16) にする（CLAUDE.md の約束）。"
  (ecase dtype
    (:f32 'single-float)
    (:f64 'double-float)
    ((:bf16 :f16) '(unsigned-byte 16))))

(defun %u32->s32 (bits)
  "符号なし32bit整数を、SBCL の make-single-float が期待する符号付き表現にする。"
  (if (>= bits #x80000000)
      (- bits #x100000000)
      bits))

(defun %single-float-bits (x)
  (logand (sb-kernel:single-float-bits x) #xFFFFFFFF))

(defun %f32-bits->bf16-bits (bits)
  "single-float のビット列の上位16bit（符号・指数・仮数の上位7bit）を切り出す。

丸めはせず切り捨てる。テスト用の配列生成にだけ使うので、これで十分。"
  (ldb (byte 16 16) bits))

(defun %bf16-bits->f32 (bf16-bits)
  (sb-kernel:make-single-float (%u32->s32 (ash bf16-bits 16))))

(defun %f32-bits->f16-bits (bits)
  "single-float のビット列を、f16 (IEEE binary16) のビット列に変換する。

丸めはせず切り捨てる。nabla のテストで使う値は絶対値がおおむね 1 以下
なので、オーバーフロー・アンダーフローの精密な扱いは必要ない。"
  (let* ((sign (ldb (byte 1 31) bits))
         (exp32 (ldb (byte 8 23) bits))
         (mant32 (ldb (byte 23 0) bits))
         (exp16 (+ (- exp32 127) 15)))
    (cond
      ((and (zerop exp32) (zerop mant32))
       (ash sign 15))
      ((<= exp16 0)
       (ash sign 15))
      ((>= exp16 31)
       (logior (ash sign 15) (ash 30 10) #x3FF))
      (t
       (logior (ash sign 15) (ash exp16 10) (ash mant32 -13))))))

(defun %f16-bits->f32 (f16-bits)
  "F16-BITS (16bit の f16 ビット列) を single-float に戻す。

指数部 (exp16) が全部0のときは非正規化数（仮数部が0なら ±0、非0なら
value = mantissa * 2^-24。f16 は指数バイアス15、仮数10bit なので、
正規化数の最小指数 2^-14 に対し、仮数の重みは 2^-14 / 2^10 = 2^-24）。
正規化数と同じ「暗黙の先頭1ビットがある」式をそのまま使うと、非正規化数
を桁違いに大きい値へデコードしてしまう。"
  (let* ((sign (ldb (byte 1 15) f16-bits))
         (exp16 (ldb (byte 5 10) f16-bits))
         (mant16 (ldb (byte 10 0) f16-bits)))
    (cond
      ((and (zerop exp16) (zerop mant16))
       (if (zerop sign) 0.0f0 -0.0f0))
      ((zerop exp16)
       (let ((magnitude (* mant16 (expt 2.0d0 -24))))
         (coerce (if (zerop sign) magnitude (- magnitude)) 'single-float)))
      (t
       (let* ((exp32 (+ (- exp16 15) 127))
              (mant32 (ash mant16 13))
              (bits32 (logior (ash sign 31) (ash exp32 23) mant32)))
         (sb-kernel:make-single-float (%u32->s32 bits32)))))))

(defun dtype-value (dtype value)
  "DOUBLE-FLOAT の VALUE を、DTYPE の配列要素として格納する値に変換する。"
  (ecase dtype
    (:f64 (coerce value 'double-float))
    (:f32 (coerce value 'single-float))
    (:bf16 (%f32-bits->bf16-bits (%single-float-bits (coerce value 'single-float))))
    (:f16 (%f32-bits->f16-bits (%single-float-bits (coerce value 'single-float))))))

(defun decode-element (dtype value)
  "配列に格納されている VALUE を、比較用の DOUBLE-FLOAT に戻す。"
  (ecase dtype
    ((:f32 :f64) (coerce value 'double-float))
    (:bf16 (coerce (%bf16-bits->f32 value) 'double-float))
    (:f16 (coerce (%f16-bits->f32 value) 'double-float))))

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

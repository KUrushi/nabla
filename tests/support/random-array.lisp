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
  (let* ((sign (ldb (byte 1 15) f16-bits))
         (exp16 (ldb (byte 5 10) f16-bits))
         (mant16 (ldb (byte 10 0) f16-bits)))
    (if (and (zerop exp16) (zerop mant16))
        (if (zerop sign) 0.0f0 -0.0f0)
        (let* ((exp32 (+ (- exp16 15) 127))
               (mant32 (ash mant16 13))
               (bits32 (logior (ash sign 31) (ash exp32 23) mant32)))
          (sb-kernel:make-single-float (%u32->s32 bits32))))))

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

(defun make-random-array (spec &key (seed 0) (domain :any))
  "SPEC (array-spec) と SEED から決定的な配列を作る。

SPEC の shape がそのまま array-dimensions になり、dtype に応じた
element-type を持つ。DOMAIN は :any (既定) / :positive / :unit。"
  (let* ((shape (array-spec-shape spec))
         (dtype (array-spec-dtype spec))
         (state (sb-ext:seed-random-state seed))
         (array (make-array shape :element-type (element-type-for-dtype dtype))))
    (dotimes (i (array-total-size array) array)
      (setf (row-major-aref array i)
            (dtype-value dtype (%random-domain-value state domain))))))

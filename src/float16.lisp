;;;; bf16 / f16 のビット列（(unsigned-byte 16)）と single-float の変換。
;;;;
;;;; CLAUDE.md の約束どおり bf16 / f16 は (unsigned-byte 16) の配列にビット
;;;; 列をそのまま持つ。ここは、その表現と single-float との間を変換する
;;;; 唯一の場所（issue #38）。すべて内部（export しない）で、eager 実装
;;;; （defprimitive の :eager）から使われる。
;;;;
;;;; single-float -> 16bit の丸めは最近接偶数丸め（RNE†、用語集参照）。
;;;; NaN / 無限大 / 非正規化数 / オーバーフローの扱いは docs/glossary.md
;;;; と各関数の docstring を参照。分類は常にビット演算で行い、NaN を含む
;;;; 浮動小数点の比較・演算は避ける（SBCL は既定で :invalid トラップが
;;;; 有効なため）。

(in-package #:nabla)

(defun %single-float-bits (x)
  "SINGLE-FLOAT X のビット列を符号なし32bit整数として返す。"
  (logand (sb-kernel:single-float-bits x) #xFFFFFFFF))

(defun %make-single-float (bits)
  "符号なし32bit整数 BITS から SINGLE-FLOAT を作る。"
  (sb-kernel:make-single-float
   (if (>= bits #x80000000) (- bits #x100000000) bits)))

(defun %floor-log2 (r)
  "正の有理数 R について、2^E <= R < 2^(E+1) を満たす整数 E を返す
（floor(log2 R)）。浮動小数点への変換を経由せず、有理数のまま正確に
計算する。"
  (let ((e (- (integer-length (numerator r)) (integer-length (denominator r)))))
    (loop while (< r (expt 2 e)) do (decf e))
    (loop while (>= r (expt 2 (1+ e))) do (incf e))
    e))

(defun %round-half-even (num den)
  "非負整数 NUM/DEN（DEN > 0）を最近接偶数丸め（RNE）で整数に丸める。"
  (multiple-value-bind (q r) (truncate num den)
    (let ((2r (* 2 r)))
      (cond ((< 2r den) q)
            ((> 2r den) (1+ q))
            (t (if (evenp q) q (1+ q)))))))

(defun %round-scaled (r shift)
  "非負有理数 R に 2^SHIFT を掛けた値を、RNE で整数に丸める。"
  (let ((num (numerator r)) (den (denominator r)))
    (if (>= shift 0)
        (%round-half-even (* num (ash 1 shift)) den)
        (%round-half-even num (* den (ash 1 (- shift)))))))

(defun %fp16-inf-magnitude (mantissa-bits exponent-bits)
  "符号を除いた無限大のビット列（指数部が全1、仮数部が0）。"
  (ash (1- (ash 1 exponent-bits)) mantissa-bits))

(defun %fp16-nan-magnitude (mantissa-bits exponent-bits)
  "符号を除いた canonical quiet NaN のビット列（指数部が全1、仮数部の
最上位ビットだけが1）。"
  (logior (%fp16-inf-magnitude mantissa-bits exponent-bits)
          (ash 1 (1- mantissa-bits))))

(defun %round-normal-magnitude (r e0 mantissa-bits)
  "正規化数として R を仮数 MANTISSA-BITS ビットに RNE で丸める。E0 は
(floor (log2 r))。丸めで仮数が桁上がりして指数が1つ増える場合は、その
補正後の指数を返す。(values 指数 仮数フィールド)。"
  (let* ((shift (- mantissa-bits e0))
         (q (%round-scaled r shift))
         (carry-value (ash 1 (1+ mantissa-bits))))
    (if (= q carry-value)
        (values (1+ e0) 0)
        (values e0 (- q (ash 1 mantissa-bits))))))

(defun %round-subnormal-magnitude (r min-normal-exponent mantissa-bits)
  "非正規化数の範囲（R < 2^MIN-NORMAL-EXPONENT）で R を RNE で丸める。
丸めた結果がちょうど最小の正規化数に達したら、正規化数として返す
（(values 1 0)）。(values 指数部の biased 値 仮数フィールド)。"
  (let* ((shift (- mantissa-bits min-normal-exponent))
         (q (%round-scaled r shift)))
    (if (= q (ash 1 mantissa-bits))
        (values 1 0)
        (values 0 q))))

(defun %encode-magnitude (r mantissa-bits exponent-bits bias)
  "非負の有理数 R（有限）の絶対値を、符号を除いた MANTISSA-BITS +
EXPONENT-BITS ビットの浮動小数点表現に RNE で変換する。丸めた結果が
最大有限値を超えたら無限大のビット列にする。"
  (if (zerop r)
      0
      (let* ((max-normal-exponent (- (- (ash 1 exponent-bits) 2) bias))
             (min-normal-exponent (- 1 bias))
             (e0 (%floor-log2 r)))
        (if (>= e0 min-normal-exponent)
            (multiple-value-bind (e mantissa) (%round-normal-magnitude r e0 mantissa-bits)
              (if (> e max-normal-exponent)
                  (%fp16-inf-magnitude mantissa-bits exponent-bits)
                  (logior (ash (+ e bias) mantissa-bits) mantissa)))
            (multiple-value-bind (biased-exp mantissa)
                (%round-subnormal-magnitude r min-normal-exponent mantissa-bits)
              (logior (ash biased-exp mantissa-bits) mantissa))))))

(defun %single-float->fp16-bits (x mantissa-bits exponent-bits bias)
  "SINGLE-FLOAT X を、MANTISSA-BITS 仮数ビット・EXPONENT-BITS 指数ビット・
指数バイアス BIAS の16bit浮動小数点のビット列に変換する（bf16 / f16 の
共通実装。RNE、NaN は符号を保った canonical quiet NaN、無限大・
オーバーフローは符号付き無限大、±0 は符号を保つ）。分類はすべてビット
演算で行い、X に対する浮動小数点の比較・演算はしない。"
  (let* ((bits (%single-float-bits x))
         (sign (ldb (byte 1 31) bits))
         (exp32 (ldb (byte 8 23) bits))
         (mant32 (ldb (byte 23 0) bits))
         (width (+ mantissa-bits exponent-bits)))
    (cond
      ((and (= exp32 #xFF) (/= mant32 0))
       (logior (ash sign width) (%fp16-nan-magnitude mantissa-bits exponent-bits)))
      ((= exp32 #xFF)
       (logior (ash sign width) (%fp16-inf-magnitude mantissa-bits exponent-bits)))
      ((and (zerop exp32) (zerop mant32))
       (ash sign width))
      (t
       (logior (ash sign width)
               (%encode-magnitude (abs (rational x)) mantissa-bits exponent-bits bias))))))

(defun bf16-bits->single-float (bits)
  "BITS ((unsigned-byte 16)、bf16 のビット列) を SINGLE-FLOAT に正確に
変換する。bf16 は指数部の幅が f32 と同じ8bitで、仮数部の上位7bitだけを
持つ形式なので、下位16bitを0で埋めるだけで常に正確な変換になる
（±0・非正規化数・無限大・NaN を含む）。"
  (%make-single-float (ash bits 16)))

(defun single-float->bf16-bits (x)
  "SINGLE-FLOAT X を bf16 のビット列に RNE で変換する。"
  (%single-float->fp16-bits x 7 8 127))

(defun %f16-subnormal->single-float-bits (sign mant16)
  "F16 の非正規化数（value = MANT16 * 2^-24）を SINGLE-FLOAT のビット列
（32bit）にする。SIGN は符号ビット（0 か 1）。MANT16 が0なら±0になる。"
  (let ((magnitude (* mant16 (expt 2.0d0 -24))))
    (logior (ash sign 31)
            (%single-float-bits (coerce magnitude 'single-float)))))

(defun f16-bits->single-float (bits)
  "BITS ((unsigned-byte 16)、f16 (IEEE binary16) のビット列) を
SINGLE-FLOAT に正確に変換する。非正規化数（指数部が全0）は
value = 仮数 * 2^-24。無限大・NaN（指数部が全1）はそれぞれ SINGLE-FLOAT
の無限大・NaN にする。"
  (let* ((sign (ldb (byte 1 15) bits))
         (exp16 (ldb (byte 5 10) bits))
         (mant16 (ldb (byte 10 0) bits)))
    (cond
      ((and (zerop exp16) (zerop mant16))
       (if (zerop sign) 0.0f0 -0.0f0))
      ((zerop exp16)
       (%make-single-float
        (%f16-subnormal->single-float-bits sign mant16)))
      ((= exp16 #x1F)
       (%make-single-float (logior (ash sign 31) (ash #xFF 23) (ash mant16 13))))
      (t
       (%make-single-float
        (logior (ash sign 31) (ash (+ (- exp16 15) 127) 23) (ash mant16 13)))))))

(defun single-float->f16-bits (x)
  "SINGLE-FLOAT X を f16 (IEEE binary16) のビット列に RNE で変換する。"
  (%single-float->fp16-bits x 10 5 15))

(defun decode-float16 (bits dtype)
  "BITS ((unsigned-byte 16)) を DTYPE（:bf16 または :f16）として
SINGLE-FLOAT に変換する。"
  (ecase dtype
    (:bf16 (bf16-bits->single-float bits))
    (:f16 (f16-bits->single-float bits))))

(defun encode-float16 (x dtype)
  "SINGLE-FLOAT X を DTYPE（:bf16 または :f16）のビット列
((unsigned-byte 16)) に RNE で変換する。"
  (ecase dtype
    (:bf16 (single-float->bf16-bits x))
    (:f16 (single-float->f16-bits x))))

(defun decode-float16-array (array dtype)
  "(unsigned-byte 16) の ARRAY（DTYPE は :bf16 または :f16）を、同じ shape
の SINGLE-FLOAT の配列に変換する。ARRAY を破壊せず、新しい配列を返す。
rank 0 の配列も扱える。"
  (let ((result (make-array (array-dimensions array) :element-type 'single-float)))
    (dotimes (i (array-total-size array) result)
      (setf (row-major-aref result i)
            (decode-float16 (row-major-aref array i) dtype)))))

(defun encode-float16-array (array dtype)
  "SINGLE-FLOAT の ARRAY を、DTYPE（:bf16 または :f16）のビット列を持つ
(unsigned-byte 16) の配列に RNE で変換する。ARRAY を破壊せず、新しい
配列を返す。rank 0 の配列も扱える。"
  (let ((result (make-array (array-dimensions array) :element-type '(unsigned-byte 16))))
    (dotimes (i (array-total-size array) result)
      (setf (row-major-aref result i)
            (encode-float16 (row-major-aref array i) dtype)))))

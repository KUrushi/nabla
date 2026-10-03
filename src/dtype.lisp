;;;; dtype: nabla が扱う要素型のタグと、Lisp の配列要素型との対応表。
;;;;
;;;; ここが唯一の場所（CLAUDE.md 設計上の約束）。実行系ごとの要素型
;;;; コード（実行系を実装するシステムの runtime が持つ対応表）は、
;;;; これとは別の1か所（実装するシステムの中）に持つ。core は実行系
;;;; 固有の名前を知らないため（#9 の backend プロトコルの設計）、
;;;; 2つの表を1つにまとめることはできない。

(in-package #:nabla)

(deftype dtype ()
  "nabla が扱う要素型のタグ。:f32 / :f64 / :bf16 / :f16 / :i1 / :i32 /
:u32 / :u64 のいずれか。
:i1 は真偽値（1ビット）で、compare の出力・select の条件に使う
（issue #37）。:i32 / :u32 / :u64 は整数（符号付き32ビット、符号なし32ビット、
符号なし64ビット。issue #126。scan のループカウンタと PRNG のキー・乱数ビット
が使う。:u64 は THREE_FRY の状態 ui64[2] のために足した）。実行系がデバイス上の表現を持たない dtype を TO-DEVICE に
渡すと UNSUPPORTED-DTYPE が signal される（実行系との要素型の対応は各実行系
の実装が決める。core はその対応を知らない）。"
  '(member :f32 :f64 :bf16 :f16 :i1 :i32 :u32 :u64))

(define-condition dtype-mismatch (error)
  ((element-type :initarg :element-type :reader dtype-mismatch-element-type)
   (dtype :initarg :dtype :reader dtype-mismatch-dtype))
  (:report
   (lambda (condition stream)
     (format stream "配列の要素型 ~S と dtype ~S が一致しない。"
             (dtype-mismatch-element-type condition)
             (dtype-mismatch-dtype condition))))
  (:documentation
   "配列の実際の要素型と、指定した（または推論しようとした）dtype タグが
矛盾するときに ARRAY-DTYPE が signal する。ELEMENT-TYPE は
ARRAY-ELEMENT-TYPE の返り値、DTYPE は呼び出し時に渡された dtype キーワード
（渡していなければ NIL）。"))

(defun dtype-element-type (dtype)
  "DTYPE に対応する Common Lisp の配列要素型を返す。

:f32 → SINGLE-FLOAT、:f64 → DOUBLE-FLOAT、:bf16 / :f16 → (UNSIGNED-BYTE 16)
（bf16 / f16 はビット列をそのまま持つ。CLAUDE.md の約束）、:i1 → BIT、
:i32 → (SIGNED-BYTE 32)、:u32 → (UNSIGNED-BYTE 32)、:u64 → (UNSIGNED-BYTE 64)。"
  (check-type dtype dtype)
  (ecase dtype
    (:f32 'single-float)
    (:f64 'double-float)
    ((:bf16 :f16) '(unsigned-byte 16))
    (:i1 'bit)
    (:i32 '(signed-byte 32))
    (:u32 '(unsigned-byte 32))
    (:u64 '(unsigned-byte 64))))

(defun dtype-byte-width (dtype)
  "DTYPE の1要素あたりのバイト数を返す。:f32 → 4、:f64 → 8、
:bf16 / :f16 → 2、:i1 → 1、:i32 / :u32 → 4、:u64 → 8。"
  (check-type dtype dtype)
  (ecase dtype
    (:f32 4)
    (:f64 8)
    ((:bf16 :f16) 2)
    (:i1 1)
    ((:i32 :u32) 4)
    (:u64 8)))

(defun array-dtype (array &optional dtype)
  "ARRAY の要素型から dtype キーワードを決めて返す。

SINGLE-FLOAT の配列は :f32、DOUBLE-FLOAT の配列は :f64、BIT の配列は :i1、
(SIGNED-BYTE 32) の配列は :i32、(UNSIGNED-BYTE 32) の配列は :u32、
(UNSIGNED-BYTE 64) の配列は :u64 と決まる。(UNSIGNED-BYTE 16) の配列は :bf16 と :f16 のどちらとも解釈できる
ため、DTYPE で必ずどちらかを指定する必要がある。

DTYPE を渡した場合、ARRAY の要素型から決まる dtype と食い違っていれば
（:f32 の配列に :f64 を渡す、(unsigned-byte 16) の配列に :f32 を渡す、
(unsigned-byte 16) の配列に DTYPE を渡さない、BIT の配列に :i1 以外を
渡す、など）、DTYPE-MISMATCH を signal する。サポートしない要素型（上の
4通り以外）も DTYPE-MISMATCH になる。"
  (let ((element-type (array-element-type array)))
    (flet ((signal-mismatch ()
             (error 'dtype-mismatch :element-type element-type :dtype dtype))
           (type= (a b)
             ;; SUBTYPEP は片方向にしか調べないため、(unsigned-byte 8) や
             ;; BIT のような (unsigned-byte 16) の真の部分型、あるいは
             ;; 空型 NIL（SUBTYPEP NIL SINGLE-FLOAT は T）まで一致した
             ;; ことにしてしまう。両方向の SUBTYPEP で厳密な型の一致を見る。
             (and (subtypep a b) (subtypep b a))))
      (cond
        ((type= element-type 'single-float)
         (if (and dtype (not (eq dtype :f32))) (signal-mismatch) :f32))
        ((type= element-type 'double-float)
         (if (and dtype (not (eq dtype :f64))) (signal-mismatch) :f64))
        ((type= element-type '(unsigned-byte 16))
         (if (member dtype '(:bf16 :f16)) dtype (signal-mismatch)))
        ((type= element-type 'bit)
         (if (and dtype (not (eq dtype :i1))) (signal-mismatch) :i1))
        ((type= element-type '(signed-byte 32))
         (if (and dtype (not (eq dtype :i32))) (signal-mismatch) :i32))
        ((type= element-type '(unsigned-byte 32))
         (if (and dtype (not (eq dtype :u32))) (signal-mismatch) :u32))
        ((type= element-type '(unsigned-byte 64))
         (if (and dtype (not (eq dtype :u64))) (signal-mismatch) :u64))
        (t (signal-mismatch))))))

(defun integer-dtype-p (dtype)
  "DTYPE が整数の dtype（:I32 :U32 :U64。:I1 は含まない）なら真を返す
（issue #126）。"
  (and (member dtype '(:i32 :u32 :u64)) t))

(defun integer-dtype-bits (dtype)
  "整数 dtype（:I32 :U32 :U64）のビット幅を返す。"
  (ecase dtype ((:i32 :u32) 32) (:u64 64)))

(defun wrap-integer (value dtype)
  "整数 VALUE を整数 DTYPE の範囲に2の補数（符号なしは法 2^n）で折り返す。
StableHLO の整数演算のオーバーフローと同じ挙動（issue #126）。"
  (let ((bits (integer-dtype-bits dtype)))
    (if (eq dtype :i32)
        (let ((u (ldb (byte bits 0) value)))
          (if (logbitp (1- bits) u) (- u (ash 1 bits)) u))
        (ldb (byte bits 0) value))))

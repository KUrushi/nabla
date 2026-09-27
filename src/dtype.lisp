;;;; dtype: nabla が扱う要素型のタグと、Lisp の配列要素型との対応表。
;;;;
;;;; ここが唯一の場所（CLAUDE.md 設計上の約束）。実行系ごとの要素型
;;;; コード（実行系を実装するシステムの runtime が持つ対応表）は、
;;;; これとは別の1か所（実装するシステムの中）に持つ。core は実行系
;;;; 固有の名前を知らないため（#9 の backend プロトコルの設計）、
;;;; 2つの表を1つにまとめることはできない。

(in-package #:nabla)

(deftype dtype ()
  "nabla が扱う要素型のタグ。:f32 / :f64 / :bf16 / :f16 / :i1 のいずれか。
:i1 は真偽値（1ビット）で、compare の出力・select の条件に使う
（issue #37）。フェーズ1では :i1 の配列を TO-DEVICE に渡すと
UNSUPPORTED-DTYPE が signal される（実行系との要素型の対応は各実行系の
実装が決める。core はその対応を知らない）。"
  '(member :f32 :f64 :bf16 :f16 :i1))

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
（bf16 / f16 はビット列をそのまま持つ。CLAUDE.md の約束）、:i1 → BIT。"
  (check-type dtype dtype)
  (ecase dtype
    (:f32 'single-float)
    (:f64 'double-float)
    ((:bf16 :f16) '(unsigned-byte 16))
    (:i1 'bit)))

(defun dtype-byte-width (dtype)
  "DTYPE の1要素あたりのバイト数を返す。:f32 → 4、:f64 → 8、
:bf16 / :f16 → 2、:i1 → 1。"
  (check-type dtype dtype)
  (ecase dtype
    (:f32 4)
    (:f64 8)
    ((:bf16 :f16) 2)
    (:i1 1)))

(defun array-dtype (array &optional dtype)
  "ARRAY の要素型から dtype キーワードを決めて返す。

SINGLE-FLOAT の配列は :f32、DOUBLE-FLOAT の配列は :f64、BIT の配列は :i1
と決まる。(UNSIGNED-BYTE 16) の配列は :bf16 と :f16 のどちらとも解釈できる
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
        (t (signal-mismatch))))))

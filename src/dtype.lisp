;;;; dtype: nabla が扱う要素型のタグと、Lisp の配列要素型との対応表。
;;;;
;;;; ここが唯一の場所（CLAUDE.md 設計上の約束）。IREE 側の
;;;; iree_hal_element_type_t との対応は、これとは別の1か所
;;;; （nabla.iree::*element-types*、src/iree/runtime.lisp）に持つ。
;;;; nabla.iree は nabla を :use しないため、2つの表を1つにまとめることは
;;;; できない（IREE 固有の名前をコアに持ち込まないため。#9 の設計）。

(in-package #:nabla)

(deftype dtype ()
  "nabla が扱う要素型のタグ。:f32 / :f64 / :bf16 / :f16 のいずれか。"
  '(member :f32 :f64 :bf16 :f16))

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
（bf16 / f16 はビット列をそのまま持つ。CLAUDE.md の約束）。"
  (check-type dtype dtype)
  (ecase dtype
    (:f32 'single-float)
    (:f64 'double-float)
    ((:bf16 :f16) '(unsigned-byte 16))))

(defun dtype-byte-width (dtype)
  "DTYPE の1要素あたりのバイト数を返す。:f32 → 4、:f64 → 8、
:bf16 / :f16 → 2。"
  (check-type dtype dtype)
  (ecase dtype
    (:f32 4)
    (:f64 8)
    ((:bf16 :f16) 2)))

(defun array-dtype (array &optional dtype)
  "ARRAY の要素型から dtype キーワードを決めて返す。

SINGLE-FLOAT の配列は :f32、DOUBLE-FLOAT の配列は :f64 と決まる。
(UNSIGNED-BYTE 16) の配列は :bf16 と :f16 のどちらとも解釈できるため、
DTYPE で必ずどちらかを指定する必要がある。

DTYPE を渡した場合、ARRAY の要素型から決まる dtype と食い違っていれば
（:f32 の配列に :f64 を渡す、(unsigned-byte 16) の配列に :f32 を渡す、
(unsigned-byte 16) の配列に DTYPE を渡さない、など）、DTYPE-MISMATCH を
signal する。サポートしない要素型（上の3通り以外）も DTYPE-MISMATCH に
なる。"
  (let ((element-type (array-element-type array)))
    (flet ((signal-mismatch ()
             (error 'dtype-mismatch :element-type element-type :dtype dtype)))
      (cond
        ((subtypep element-type 'single-float)
         (if (and dtype (not (eq dtype :f32))) (signal-mismatch) :f32))
        ((subtypep element-type 'double-float)
         (if (and dtype (not (eq dtype :f64))) (signal-mismatch) :f64))
        ((subtypep element-type '(unsigned-byte 16))
         (if (member dtype '(:bf16 :f16)) dtype (signal-mismatch)))
        (t (signal-mismatch))))))

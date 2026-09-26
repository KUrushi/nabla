;;;; aval: 配列の形状 (shape) と dtype の組（JAX の abstract value に相当）。
;;;;
;;;; 実際のデータは持たず、形状推論やデバイス上のバッファの器の記述に使う。

(in-package #:nabla)

(defstruct (aval (:constructor %make-aval (shape dtype)))
  "配列の抽象的な値。SHAPE は非負整数のリスト、DTYPE は DTYPE 型の
キーワード。等価性は EQUALP で比べる（専用の AVAL= は export しない。
ハイラムの法則に備え、公開シンボルは最小限にする）。"
  (shape nil :type list :read-only t)
  (dtype nil :type keyword :read-only t))

(defun make-aval (shape dtype)
  "SHAPE（非負整数のリスト）と DTYPE（DTYPE 型のキーワード）から AVAL を
作る。SHAPE がリストでない、SHAPE が負の次元を含む、または DTYPE が
DTYPE 型でなければエラーを signal する。"
  (check-type shape list)
  (dolist (dim shape)
    (check-type dim (integer 0)))
  (check-type dtype dtype)
  (%make-aval shape dtype))

(defun aval-rank (aval)
  "AVAL の rank（次元数）を返す。"
  (length (aval-shape aval)))

(defun aval-size (aval)
  "AVAL が表す配列の要素数（各次元の積）を返す。rank 0 なら1。"
  (reduce #'* (aval-shape aval) :initial-value 1))

(defun aval-byte-length (aval)
  "AVAL が表す配列全体のバイト数（要素数 × dtype-byte-width）を返す。"
  (* (aval-size aval) (dtype-byte-width (aval-dtype aval))))

(defun array-aval (array &optional dtype)
  "ARRAY の array-dimensions と (array-dtype array dtype) から AVAL を作る。"
  (make-aval (array-dimensions array) (array-dtype array dtype)))

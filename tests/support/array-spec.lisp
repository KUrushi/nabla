;;;; array-spec: 配列の形状と dtype を表す仕様、および check-it の生成器。

(in-package #:nabla.tests.support)

;; check-it の def-generator は同じ名前で defclass するので、構造体の型名
;; そのものは array-spec% にし、アクセサだけ :conc-name で array-spec- に
;; そろえる。ARRAY-SPEC というシンボルは、下の def-generator が定義する
;; ジェネレータのクラス名として使う。
(defstruct (array-spec% (:conc-name array-spec-)
                        (:constructor make-array-spec (shape dtype)))
  "配列の形状 (SHAPE, フィクスナムのリスト) と DTYPE (キーワード) の組。"
  (shape nil :type list :read-only t)
  (dtype nil :type keyword :read-only t))

(defun array-spec-rank (spec)
  "SPEC の形状の次元数（rank）を返す。"
  (length (array-spec-shape spec)))

(check-it:def-generator array-spec (&key (dtypes *dtypes*) (max-rank 4) (max-dim 8))
  "check-it の named generator。rank 0..MAX-RANK、各次元 1..MAX-DIM、
DTYPES の中から選んだ dtype を持つ ARRAY-SPEC を作る。

check-it の generator DSL (integer / tuple / map / chain など) は固定の
個数のサブジェネレータしか書けないので、rank ごとに違う個数の次元を
作る部分だけは、check-it が公開しているジェネレータクラス
(int-generator / tuple-generator / mapped-generator / chained-generator)
を直接組み立てて書く。"
  (make-instance 'check-it:chained-generator
                 :pre-generators
                 (list (make-instance 'check-it:int-generator
                                      :lower-limit 0
                                      :upper-limit max-rank))
                 :generator-function
                 (lambda (rank)
                   (make-instance
                    'check-it:mapped-generator
                    :sub-generators
                    (list (make-instance
                           'check-it:tuple-generator
                           :sub-generators
                           (loop repeat rank
                                 collect (make-instance 'check-it:int-generator
                                                        :lower-limit 1
                                                        :upper-limit max-dim)))
                          (make-instance 'check-it:int-generator
                                        :lower-limit 0
                                        :upper-limit (1- (length dtypes))))
                    :mapping
                    (lambda (shape dtype-index)
                      (make-array-spec shape (nth dtype-index dtypes)))))))

;;;; array-spec: 配列の形状と dtype を表す仕様、および check-it の生成器。

(in-package #:nabla.tests.support)

;; check-it の def-generator は同じ名前で defclass するので、構造体の型名
;; そのものは array-spec% にし、アクセサだけ :conc-name で array-spec- に
;; そろえる。ARRAY-SPEC というシンボルは、下の def-generator が定義する
;; ジェネレータのクラス名として使う（型名は ARRAY-SPEC% のままなので、
;; (typep x 'array-spec) は真にならない点に注意）。
;;
;; :constructor を2つ持たせているのは、check-it が失敗例を保存すると
;; き `(format nil "~S" value)` で #S(ARRAY-SPEC% :SHAPE ... :DTYPE ...)
;; という表記を使い、regression ファイルの LOAD 時に READ-FROM-STRING
;; で読み戻すため。SBCL の #S リーダーはキーワード引数の（BOA でない）
;; default constructor を要求するので、BOA constructor (MAKE-ARRAY-SPEC)
;; だけでは `The ... structure does not have a default constructor.`
;; エラーで読めない。2つ目の %MAKE-ARRAY-SPEC-KW はどこからも直接呼ばな
;; いが、これがあることで #S の読み書きが可能になる。
(defstruct (array-spec% (:conc-name array-spec-)
                        (:constructor make-array-spec (shape dtype))
                        (:constructor %make-array-spec-kw))
  "配列の形状 (SHAPE, フィクスナムのリスト) と DTYPE (キーワード) の組。"
  (shape nil :type list :read-only t)
  (dtype nil :type keyword :read-only t))

(defun array-spec-rank (spec)
  "SPEC の形状の次元数（rank）を返す。"
  (length (array-spec-shape spec)))

;; check-it の named generator。rank 0..MAX-RANK、各次元 1..MAX-DIM、
;; DTYPES の中から選んだ dtype を持つ ARRAY-SPEC を作る。
;;
;; check-it の generator DSL (integer / tuple / map / chain など) は固定の
;; 個数のサブジェネレータしか書けないので、rank ごとに違う個数の次元を
;; 作る部分だけは、check-it が公開しているジェネレータクラス
;; (tuple-generator / mapped-generator / chained-generator) を直接組み立
;; てて書く。整数の範囲を選ぶ部分には check-it 組み込みの int-generator
;; ではなく UNIFORM-INTEGER-GENERATOR (uniform-generator.lisp) を使う。
;; int-generator は check-it::*size*（既定10）で lo/hi をクランプしてし
;; まうため、MAX-RANK や MAX-DIM に既定の 4 / 8 より大きい値を渡すと、
;; 呼び出し側に何のエラーも出さずに実際の rank / 次元の上限が10で頭打ち
;; になってしまう（.claude/skills/nabla-testing/references/properties.md
;; 参照）。
;;
;; def-generator の &body はそのまま generate メソッドの本体に展開され、
;; defun のような docstring の特別扱いはしない（先頭に文字列を置いても
;; 無害な式として評価されるだけで捨てられる）。そのため説明はここに
;; コメントとして書く。
(check-it:def-generator array-spec (&key (dtypes *dtypes*) (max-rank 4) (max-dim 8))
  (make-instance 'check-it:chained-generator
                 :pre-generators
                 (list (make-uniform-integer-generator 0 max-rank))
                 :generator-function
                 (lambda (rank)
                   (make-instance
                    'check-it:mapped-generator
                    :sub-generators
                    (list (make-instance
                           'check-it:tuple-generator
                           :sub-generators
                           (loop repeat rank
                                 collect (make-uniform-integer-generator 1 max-dim)))
                          (make-uniform-integer-generator 0 (1- (length dtypes))))
                    :mapping
                    (lambda (shape dtype-index)
                      (make-array-spec shape (nth dtype-index dtypes)))))))

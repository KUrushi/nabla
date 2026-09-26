;;;; uniform-integer / uniform-real: check-it::*size* に縛られない一様な
;;;; 整数・実数の check-it generator。
;;;;
;;;; check-it 組み込みの (integer lo hi) / (real lo hi) は、指定した lo/hi
;;;; に関係なく check-it::*size*（既定10）に値をクランプする
;;;; (int-generator-function / real-generator-function 参照。lo/hi を
;;;; 指定していても内部で (min (abs limit) *size*) を取る)。そのため
;;;; たとえば (generator (integer 0 1023)) は実際には 0..10 の値しか
;;;; 生成しない。この事実は check-it のドキュメントにはなく、生成された
;;;; 値の分布を実際に見て初めて気づける類の落とし穴で、この PR で書いた
;;;; PBT がまさにこれに引っかかっていた（詳しくは
;;;; .claude/skills/nabla-testing/references/properties.md を参照）。
;;;;
;;;; ここで定義する uniform-integer / uniform-real は、check-it の
;;;; generator プロトコル（GENERATE / SHRINK）だけを自前で実装し、
;;;; check-it 組み込みの int-generator-function / real-generator-function
;;;; を経由しないことで、この *size* によるクランプを回避する。

(in-package #:nabla.tests.support)

(defclass %uniform-integer-generator (check-it:generator)
  ((lo :initarg :lo :reader %uniform-integer-lo)
   (hi :initarg :hi :reader %uniform-integer-hi))
  (:documentation
   "LO..HI（両端を含む）の整数を、check-it::*size* に関係なく一様に生成する。"))

(defmethod check-it:generate ((generator %uniform-integer-generator))
  (+ (%uniform-integer-lo generator)
     (random (1+ (- (%uniform-integer-hi generator) (%uniform-integer-lo generator))))))

(defmethod check-it:shrink ((generator %uniform-integer-generator) test)
  "check-it 組み込みの int-generator と同じく、[LO, HI] の範囲内で0に
一番近い反例（TEST が偽を返す値）まで縮小する（0 が範囲外なら LO に
向けて縮小する）。

一様に値を選ぶことと、0 に向けて縮小できないことは別の話。check-it
組み込みの int-generator も一様に選ぶが、SHRINK は
(defmethod shrink ((value integer) test) (shrink-int value test 0 value))
という、値そのものに対する汎用の縮小（0 に向けた二分探索）に委ねて
いる（shrink.lisp）。ここでも同じ SHRINK を使い、範囲外の値は
「TEST は真（=性質は成り立つ、反例ではない）」として扱うことで、
縮小後の値が必ず [LO, HI] に収まるようにする（int-generator の
int-shrinker-predicate と同じ考え方。generators.lisp 参照）。"
  (let ((lo (%uniform-integer-lo generator))
        (hi (%uniform-integer-hi generator)))
    (setf (check-it:cached-value generator)
          (check-it:shrink (check-it:cached-value generator)
                            (lambda (x) (or (< x lo) (> x hi) (funcall test x)))))))

(defun make-uniform-integer-generator (lo hi)
  "LO..HI（両端を含む）の整数を一様に生成する check-it generator インスタンスを作る。"
  (assert (<= lo hi) (lo hi)
          "make-uniform-integer-generator: LO (~A) が HI (~A) より大きい" lo hi)
  (make-instance '%uniform-integer-generator :lo lo :hi hi))

(check-it:def-generator uniform-integer (&key lo hi)
  ;; (generator (uniform-integer :lo LO :hi HI)) の形で使う check-it の
  ;; named generator。check-it 組み込みの (integer lo hi) と違い、
  ;; check-it::*size*（既定10）でクランプされず、LO..HI 全域から一様に
  ;; 整数を選ぶ。
  (make-uniform-integer-generator lo hi))

(defclass %uniform-real-generator (check-it:generator)
  ((lo :initarg :lo :reader %uniform-real-lo)
   (hi :initarg :hi :reader %uniform-real-hi))
  (:documentation
   "LO..HI（LO を含み HI を含まない）の DOUBLE-FLOAT を、check-it::*size*
に関係なく一様に生成する。"))

(defmethod check-it:generate ((generator %uniform-real-generator))
  (+ (%uniform-real-lo generator)
     (* (random 1.0d0) (- (%uniform-real-hi generator) (%uniform-real-lo generator)))))

(defmethod check-it:shrink ((generator %uniform-real-generator) test)
  "check-it 組み込みの real-generator
(defmethod shrink ((value real) test) (declare (ignore test)) value)
と同じく、縮小はせず、失敗時の値をそのまま返す。UNIFORM-INTEGER と
違い、これは「一様に選ぶから縮小できない」という意味ではなく、実数
は離散的な探索空間ではないので check-it に汎用の縮小アルゴリズムが
無い、というだけ（shrink.lisp の real 用の SHRINK メソッド参照）。"
  (declare (ignore test))
  (check-it:cached-value generator))

(defun make-uniform-real-generator (lo hi)
  "LO..HI（LO を含み HI を含まない）の DOUBLE-FLOAT を一様に生成する
check-it generator インスタンスを作る。"
  (assert (<= lo hi) (lo hi)
          "make-uniform-real-generator: LO (~A) が HI (~A) より大きい" lo hi)
  (make-instance '%uniform-real-generator :lo lo :hi hi))

(check-it:def-generator uniform-real (&key lo hi)
  ;; (generator (uniform-real :lo LO :hi HI)) の形で使う。UNIFORM-INTEGER
  ;; と同じ理由で、check-it 組み込みの (real lo hi) の *size* によるクラ
  ;; ンプを避けたいときに使う。
  (make-uniform-real-generator lo hi))

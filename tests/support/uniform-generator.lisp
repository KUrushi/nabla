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
  "整数だが、区間全体を意図して一様に選んでいる（check-it 組み込みの
int-generator のように、0 に向けて縮小する探索空間ではない）ので、
real-generator と同じく縮小はせず、失敗時の値をそのまま返す。"
  (declare (ignore test))
  (check-it:cached-value generator))

(defun make-uniform-integer-generator (lo hi)
  "LO..HI（両端を含む）の整数を一様に生成する check-it generator インスタンスを作る。"
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
  "check-it 組み込みの real-generator と同じく、連続的な探索空間なので
縮小はせず、失敗時の値をそのまま返す。"
  (declare (ignore test))
  (check-it:cached-value generator))

(defun make-uniform-real-generator (lo hi)
  "LO..HI（LO を含み HI を含まない）の DOUBLE-FLOAT を一様に生成する
check-it generator インスタンスを作る。"
  (make-instance '%uniform-real-generator :lo lo :hi hi))

(check-it:def-generator uniform-real (&key lo hi)
  ;; (generator (uniform-real :lo LO :hi HI)) の形で使う。UNIFORM-INTEGER
  ;; と同じ理由で、check-it 組み込みの (real lo hi) の *size* によるクラ
  ;; ンプを避けたいときに使う。
  (make-uniform-real-generator lo hi))

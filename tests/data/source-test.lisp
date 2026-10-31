;;;; data-source と array-source（issue #279）。
;;;;
;;;; 守らせる性質（small）:
;;;; - 行の一致: array-source の index 番目のレコードは、元の木と同じ形（treedef=）で、
;;;;   各葉は元の葉の先頭の軸の index 番目の行（同じ element-type、先頭の軸を除いた形、
;;;;   同じビット列）。
;;;; - 長さ: source-length は葉の先頭の軸の長さ（0 も含む）。
;;;; - 範囲外: 0 未満・長さ以上・整数でない添字は :index-out-of-range の data-source-error。
;;;; - 不正な木: 葉が無い・配列でない葉・rank 0 の葉・先頭の軸の長さの食い違いは、
;;;;   理由つきの data-source-error（作った時点で）。
;;;; - 依存の方針: nabla/data が直接依存するシステムは core（nabla）だけ。

(in-package #:nabla.data.tests)

(defparameter *source-element-types*
  '(single-float double-float (unsigned-byte 16) bit (signed-byte 32) (unsigned-byte 32)
    (unsigned-byte 64) (unsigned-byte 8))
  "葉の element-type の候補。nabla の dtype に無い (unsigned-byte 8)（画像の画素）も入れる。")

;;; --- 生成 ---

(defun source-random-element (element-type state)
  (cond ((eq element-type 'single-float) (- (random 2.0 state) 1.0))
        ((eq element-type 'double-float) (- (random 2d0 state) 1d0))
        ((eq element-type 'bit) (random 2 state))
        ((equal element-type '(signed-byte 32)) (- (random (ash 1 32) state) (ash 1 31)))
        (t (random (ash 1 (second element-type)) state))))

(defun source-random-leaf (length state)
  "先頭の軸が LENGTH で、残りの軸が rank 0..2・各次元 0..3 の、ランダムな element-type の配列。"
  (let* ((element-type (nth (random (length *source-element-types*) state) *source-element-types*))
         (array (make-array (cons length (loop repeat (random 3 state) collect (random 4 state)))
                            :element-type element-type)))
    (dotimes (i (array-total-size array) array)
      (setf (row-major-aref array i) (source-random-element element-type state)))))

(defun source-random-tree (length state depth)
  "葉の先頭の軸がどれも LENGTH の、リスト・ベクタ・PYTREE-TEST-PAIR・nil・配列の入れ子の木。"
  (flet ((sub () (source-random-tree length state (1- depth))))
    (if (<= depth 0)
        (source-random-leaf length state)
        (ecase (random 6 state)
          ((0 5) (source-random-leaf length state))
          (1 (loop repeat (random 4 state) collect (sub)))
          (2 (coerce (loop repeat (random 4 state) collect (sub)) 'simple-vector))
          (3 (make-pytree-test-pair (sub) (sub) (random 3 state)))
          (4 nil)))))

(defun source-random-source-tree (seed)
  "SEED から決まる (values 木 長さ)。木は葉を1つ以上持つ。"
  (let* ((state (sb-ext:seed-random-state seed))
         (length (random 5 state))
         (tree (source-random-tree length state (random 4 state))))
    (values (if (nabla:tree-leaves tree)
                tree
                (list tree (source-random-leaf length state)))
            length)))

;;; --- オラクル ---

(defun row-matches-p (leaf index record-leaf)
  "RECORD-LEAF が LEAF の先頭の軸の INDEX 番目の行と同じ element-type・形・要素なら真。"
  (let ((row-size (array-total-size record-leaf)))
    (and (equal (array-element-type record-leaf) (array-element-type leaf))
         (equal (array-dimensions record-leaf) (rest (array-dimensions leaf)))
         (loop for j below row-size
               always (eql (row-major-aref record-leaf j)
                           (row-major-aref leaf (+ (* index row-size) j)))))))

(defun record-matches-p (tree index record)
  (and (nabla:treedef= (nabla:tree-structure tree) (nabla:tree-structure record))
       (every (lambda (leaf record-leaf) (row-matches-p leaf index record-leaf))
              (nabla:tree-leaves tree)
              (nabla:tree-leaves record))))

(defun source-error-reason (thunk)
  (handler-case (progn (funcall thunk) :no-error)
    (data-source-error (c) (data-source-error-reason c))))

(defun f32-vector (&rest values)
  (make-array (length values) :element-type 'single-float :initial-contents values))

(defmacro def-source-property (name docstring (&rest vars) generator &body body)
  "GENERATOR が作る値を VARS に束縛して BODY（真偽）を small の PBT で確かめる。"
  `(test (,name :suite :nabla.small)
     ,docstring
     (is (check-it ,generator
                   (lambda (case) (destructuring-bind ,vars case ,@body))
                   :regression-id ,name
                   :regression-file (regression-path ,(substitute #\- #\/ (string-downcase (string name)))
                                                     :package "NABLA.DATA.TESTS"))
         ,(format nil "~A の性質が成り立たなかった" name))))

;;; --- 行の一致と長さ ---

(def-source-property array-source/record-is-the-row-of-every-leaf
  "index 番目のレコードは、元の木と同じ形で、各葉の index 番目の行。"
  (seed index)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo 0 :hi 4)))
  (multiple-value-bind (tree length) (source-random-source-tree seed)
    (or (>= index length)
        (record-matches-p tree index (source-item (make-array-source tree) index)))))

(def-source-property array-source/length-is-the-leading-axis
  "source-length は葉の先頭の軸の長さ（0 も含む）。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (multiple-value-bind (tree length) (source-random-source-tree seed)
    (= length (source-length (make-array-source tree)))))

(test (array-source/record-of-a-single-array :suite :nabla.small)
  "葉が配列そのもの（木の根が葉）なら、レコードも配列。1 次元の葉の行は rank 0 の配列。"
  (let ((source (make-array-source (f32-vector 1.0 2.0 3.0))))
    (is (= 3 (source-length source)))
    (is (equal '() (array-dimensions (source-item source 2))))
    (is (= 3.0 (aref (source-item source 2))))))

;;; --- 範囲外の添字 ---

(def-source-property array-source/index-out-of-range-is-reported
  "0 未満・長さ以上の添字は :index-out-of-range。"
  (seed offset)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo 0 :hi 3)))
  (multiple-value-bind (tree length) (source-random-source-tree seed)
    (let ((source (make-array-source tree)))
      (and (eq :index-out-of-range (source-error-reason (lambda () (source-item source (+ length offset)))))
           (eq :index-out-of-range (source-error-reason (lambda () (source-item source (- -1 offset)))))))))

(test (array-source/non-integer-index-is-reported :suite :nabla.small)
  (let ((source (make-array-source (f32-vector 1.0 2.0))))
    (is (eq :index-out-of-range (source-error-reason (lambda () (source-item source 1.0)))))
    (is (eq :index-out-of-range (source-error-reason (lambda () (source-item source nil)))))))

;;; --- 不正な木 ---

(test (array-source/rejects-a-tree-without-leaves :suite :nabla.small)
  (is (eq :no-leaves (source-error-reason (lambda () (make-array-source nil)))))
  (is (eq :no-leaves (source-error-reason (lambda () (make-array-source (list nil (vector))))))))

(test (array-source/rejects-leaves-that-are-not-arrays :suite :nabla.small)
  (is (eq :bad-leaf (source-error-reason (lambda () (make-array-source (list (f32-vector 1.0) 3))))))
  (is (eq :bad-leaf (source-error-reason (lambda () (make-array-source (list (f32-vector 1.0) :x)))))))

(test (array-source/rejects-rank-zero-leaves :suite :nabla.small)
  (is (eq :rank-zero-leaf
          (source-error-reason
           (lambda () (make-array-source (list (f32-vector 1.0)
                                               (make-array '() :element-type 'single-float
                                                               :initial-element 0.0))))))))

(def-source-property array-source/rejects-leaves-of-different-lengths
  "先頭の軸の長さが違う葉を足した木は :length-mismatch。"
  (seed extra)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo 1 :hi 3)))
  (multiple-value-bind (tree length) (source-random-source-tree seed)
    (let ((other (source-random-leaf (+ length extra) (sb-ext:seed-random-state seed))))
      (and (eq :length-mismatch (source-error-reason (lambda () (make-array-source (list tree other)))))
           (eq :length-mismatch (source-error-reason (lambda () (make-array-source (list other tree)))))))))

(test (data-source-error/is-a-data-error :suite :nabla.small)
  (signals data-error (make-array-source nil)))

;;; --- 依存の方針 ---

(test (data/depends-only-on-core :suite :nabla.small)
  "nabla/data は nabla のシステムのうち core にだけ依存し、nabla/nn・nabla/iree・nabla/pjrt を
知らない（issue #279。lparallel のような外部のライブラリには依存してよい）。"
  (is (equal '("nabla")
             (remove-if-not (lambda (dependency)
                              (and (stringp dependency) (eql 0 (search "nabla" dependency))))
                            (asdf:system-depends-on (asdf:find-system "nabla/data"))))))

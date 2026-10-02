;;;; graph-recipes: check-it が var / graph 構造体を直接生成しないための
;;;; 「レシピ」形式（issue #29、u1a。u1b の read-graph のテストとも共有する）。
;;;;
;;;; check-it は失敗例を (format nil "~S" datum) で regression ファイルに書き、
;;;; read-from-string で読み戻す。EQ 同一性で結ばれた var を含む graph は、
;;;; この往復で壊れる（同じ var が別々の #S に読まれ SSA が崩れる）ので、
;;;; check-it の生成器はプレーンなリスト（レシピ）を返し、BUILD-GRAPH で
;;;; テスト本体がグラフを組み立てる。
;;;;
;;;; レシピはステップのリスト:
;;;;   (:in dtype shape)                 入力 var を1つ作る
;;;;   (:const dtype shape seed)         定数 var を1つ作る
;;;;   (:unary prim-keyword idx)         単項プリミティブを適用（%test-neg）
;;;;   (:binary prim-keyword idx idx)    二項プリミティブを適用（%test-add。
;;;;                                     GRAPH-RECIPE の :binary-prims で %test-mul も選べる）
;;;;   (:reshape idx shape)              %test-reshape を適用
;;;;   (:convert idx dtype)              %test-convert を適用
;;;;   (:reduce idx axis)                %test-reduce を適用
;;;;   (:out idx*)                       グラフの出力 var を選ぶ（末尾に1つ）
;;;;
;;;; IDX は「それまでに定義された var の番号」（0 始まり、in → const →
;;;; 各ステップの出力の順）。生成器は常にこの範囲からしか選ばないので、
;;;; use-before-def なレシピは生成しない。

(in-package #:nabla.tests)

(defun %recipe-random-shape (max-rank max-dim)
  (loop repeat (random (1+ max-rank)) collect (1+ (random max-dim))))

(defun %recipe-random-dtype (dtypes)
  (nth (random (length dtypes)) dtypes))

(defun %recipe-aval-of (avals idx)
  (nth idx avals))

(defun %recipe-matching-index (avals idx1)
  "AVALS の中から IDX1 と EQUALP な aval（shape . dtype）を持つ、IDX1 以外の
インデックスを探す。無ければ NIL。"
  (let ((target (%recipe-aval-of avals idx1)))
    (loop for i from 0
          for a in avals
          when (and (/= i idx1) (equalp a target))
            return i)))

(defun %generate-recipe (max-ops dtypes max-rank max-dim &optional (binary-prims '(:%test-add)))
  "レシピ（ステップのリスト）を1つランダムに生成して返す。少なくとも1つの
:IN ステップと、末尾に1つの :OUT ステップを持つ。"
  (let ((steps '())
        ;; AVALS は (shape . dtype) のリスト。インデックスは VARS と対応する。
        (avals '())
        (n-in (1+ (random 3))))
    (dotimes (_ n-in)
      (let ((shape (%recipe-random-shape max-rank max-dim))
            (dtype (%recipe-random-dtype dtypes)))
        (push (list :in dtype shape) steps)
        (setf avals (append avals (list (cons shape dtype))))))
    (dotimes (_ (random 3))
      (let ((shape (%recipe-random-shape max-rank max-dim))
            (dtype (%recipe-random-dtype dtypes)))
        (push (list :const dtype shape (random (expt 2 31))) steps)
        (setf avals (append avals (list (cons shape dtype))))))
    (dotimes (_ (random (1+ max-ops)))
      (let* ((n (length avals))
             (idx (random n))
             (aval (nth idx avals))
             (shape (car aval))
             (rank (length shape))
             ;; :reduce には rank >= 1 が要る。
             (choices (if (plusp rank) '(:unary :binary :reshape :convert :reduce)
                          '(:unary :binary :reshape :convert)))
             (kind (nth (random (length choices)) choices)))
        (ecase kind
          (:unary
           (push (list :unary :%test-neg idx) steps)
           (setf avals (append avals (list aval))))
          (:binary
           (let* ((idx2 (or (%recipe-matching-index avals idx) idx)))
             (push (list :binary (nth (random (length binary-prims)) binary-prims) idx idx2) steps)
             (setf avals (append avals (list aval)))))
          (:reshape
           ;; 要素数が変わらない reshape（フラット化）にする。BUILD-GRAPH が
           ;; %test-reshape の abstract-eval に渡すので、要素数さえ合っていれば
           ;; 具体的な形は問わない。
           (let ((flat (list (reduce #'* shape :initial-value 1))))
             (push (list :reshape idx flat) steps)
             (setf avals (append avals (list (cons flat (cdr aval)))))))
          (:convert
           (let ((new-dtype (%recipe-random-dtype dtypes)))
             (push (list :convert idx new-dtype) steps)
             (setf avals (append avals (list (cons shape new-dtype))))))
          (:reduce
           (let* ((axis (random rank))
                  (new-shape (append (subseq shape 0 axis) (subseq shape (1+ axis)))))
             (push (list :reduce idx axis) steps)
             (setf avals (append avals (list (cons new-shape (cdr aval))))))))))
    (push (list :out (random (length avals))) steps)
    (nreverse steps)))

(defclass %graph-recipe-generator (check-it:generator)
  ((max-ops :initarg :max-ops :reader %graph-recipe-max-ops)
   (dtypes :initarg :dtypes :reader %graph-recipe-dtypes)
   (max-rank :initarg :max-rank :reader %graph-recipe-max-rank)
   (max-dim :initarg :max-dim :reader %graph-recipe-max-dim)
   (binary-prims :initarg :binary-prims :initform '(:%test-add)
                 :reader %graph-recipe-binary-prims))
  (:documentation "GRAPH-RECIPE の named generator の実体。"))

(defmethod check-it:generate ((generator %graph-recipe-generator))
  (%generate-recipe (%graph-recipe-max-ops generator)
                     (%graph-recipe-dtypes generator)
                     (%graph-recipe-max-rank generator)
                     (%graph-recipe-max-dim generator)
                     (%graph-recipe-binary-prims generator)))

(defmethod check-it:shrink ((generator %graph-recipe-generator) test)
  ;; レシピの縮小は行わない（check-it の縮小プロトコルには cached-value を
  ;; そのまま返すだけでよい。tests/support/uniform-generator.lisp の
  ;; UNIFORM-REAL-GENERATOR と同じ考え方）。
  (declare (ignore test))
  (check-it:cached-value generator))

(check-it:def-generator graph-recipe (&key (max-ops 6) (dtypes *dtypes*) (max-rank 3) (max-dim 4)
                                         (binary-prims '(:%test-add)))
  (make-instance '%graph-recipe-generator :max-ops max-ops :dtypes dtypes
                 :max-rank max-rank :max-dim max-dim :binary-prims binary-prims))

(defun build-graph (recipe)
  "RECIPE（このファイル冒頭のレシピ形式）から NB::GRAPH を組み立てて返す。
check-graph は呼ばない（呼び出し側がレシピを壊した性質を確かめたいときに
自分で呼べるように、ここでは組み立てるだけにする）。"
  (let ((vars (make-array 0 :adjustable t :fill-pointer 0))
        (invars '())
        (constants '())
        (eqns '())
        (outvars '()))
    (flet ((push-result (eqn)
             (push eqn eqns)
             (vector-push-extend (first (nb:eqn-outvars eqn)) vars)))
      (dolist (step recipe)
        (ecase (first step)
          (:in (destructuring-bind (dtype shape) (rest step)
                 (let ((v (nb::make-var (nb:make-aval shape dtype))))
                   (vector-push-extend v vars)
                   (push v invars))))
          (:const (destructuring-bind (dtype shape seed) (rest step)
                    (let* ((array (make-random-array (make-array-spec shape dtype) :seed seed))
                           (v (nb::make-var (nb:array-aval array dtype))))
                      (vector-push-extend v vars)
                      (push (cons v array) constants))))
          (:unary (destructuring-bind (prim idx) (rest step)
                    (push-result (nb::make-eqn prim (list (aref vars idx))))))
          (:binary (destructuring-bind (prim idx1 idx2) (rest step)
                     (push-result (nb::make-eqn prim (list (aref vars idx1) (aref vars idx2))))))
          (:reshape (destructuring-bind (idx shape) (rest step)
                      (push-result (nb::make-eqn :%test-reshape (list (aref vars idx)) :shape shape))))
          (:convert (destructuring-bind (idx dtype) (rest step)
                      (push-result (nb::make-eqn :%test-convert (list (aref vars idx)) :dtype dtype))))
          (:reduce (destructuring-bind (idx axis) (rest step)
                     (push-result (nb::make-eqn :%test-reduce (list (aref vars idx)) :axis axis))))
          (:out (setf outvars (mapcar (lambda (idx) (aref vars idx)) (rest step)))))))
    (nb::make-graph (nreverse invars) (nreverse eqns) outvars (nreverse constants))))

(defun recipe-var-count (recipe)
  "RECIPE が定義する var の総数（:IN + :CONST + 演算ステップ）を返す。
BUILD-GRAPH した graph の var の定義回数を、check-graph とは独立に数える
テストのオラクルに使う。"
  (count-if (lambda (step) (member (first step) '(:in :const :unary :binary :reshape :convert :reduce)))
            recipe))

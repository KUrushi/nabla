;;;; pytree: PyTree の PBT に使う生成器（PYTREE-SPEC）と比較（PYTREE-EQUAL）（issue #195）。
;;;;
;;;; 木はリスト・SIMPLE-VECTOR・DEFPYTREE で登録したテスト用の構造体 PYTREE-TEST-PAIR を
;;;; 入れ子にし、葉は整数にする（葉を EQL で比べられるようにするため）。defmodule の構造体
;;;; （#202）ができたら、それも生成器に足す。

(in-package #:nabla.tests.support)

;; :constructor を2つ持たせるのは ARRAY-SPEC% と同じ理由（check-it が失敗例を #S(...) で
;; 保存し、regression ファイルの LOAD 時に読み戻すため。array-spec.lisp の冒頭を参照）。
(defstruct (pytree-test-pair (:constructor make-pytree-test-pair (left right tag))
                             (:constructor %make-pytree-test-pair-kw))
  "テスト用の、DEFPYTREE で登録した節。子は LEFT と RIGHT、TAG は補助データ（静的）。"
  left
  right
  (tag 0))

(nabla:defpytree pytree-test-pair
  :flatten (lambda (pair)
             (values (list (pytree-test-pair-left pair) (pytree-test-pair-right pair))
                     (pytree-test-pair-tag pair)))
  :unflatten (lambda (tag children)
               (make-pytree-test-pair (first children) (second children) tag)))

;; 葉（-100..100 の整数）か、0..3 個の子を持つリスト・ベクタ・PYTREE-TEST-PAIR。
;; 再帰の深さは check-it の *recursive-bias-decay* で抑えられる。
(check-it:def-generator pytree-spec ()
  (check-it:generator
   (or (uniform-integer :lo -100 :hi 100)
       (list (pytree-spec) :max-length 3)
       (map (lambda (children) (coerce children 'simple-vector))
            (list (pytree-spec) :max-length 3))
       (map #'make-pytree-test-pair (pytree-spec) (pytree-spec) (uniform-integer :lo 0 :hi 2)))))

(defun pytree-equal (a b)
  "A と B が同じ形（リスト・SIMPLE-VECTOR・PYTREE-TEST-PAIR とその TAG）で、葉が EQL なら真。"
  (cond
    ((and (listp a) (listp b))
     (and (= (length a) (length b)) (every #'pytree-equal a b)))
    ((and (simple-vector-p a) (simple-vector-p b))
     (and (= (length a) (length b)) (every #'pytree-equal a b)))
    ((and (pytree-test-pair-p a) (pytree-test-pair-p b))
     (and (eql (pytree-test-pair-tag a) (pytree-test-pair-tag b))
          (pytree-equal (pytree-test-pair-left a) (pytree-test-pair-left b))
          (pytree-equal (pytree-test-pair-right a) (pytree-test-pair-right b))))
    (t (eql a b))))

;;; --- 配列を葉に持つ木（jit / grad / vmap の PyTree の PBT。issue #199） ---

(defun pytree-array-tree (spec dtype &key max-leaves)
  "PYTREE-SPEC が作った整数の葉の木 SPEC の葉を、DTYPE のランダムな配列に置き換えた同じ形の木。
葉の整数 n から shape（rank 0 / 1 / 2）とシードを決めるので、同じ SPEC からは同じ木ができる。
MAX-LEAVES を渡すと、左から MAX-LEAVES 個より後ろの葉は NIL（空の部分木）に置き換える。
ランダムな木は葉が数百個になることがある。IREE を通すテストでは、(1) コンパイルが遅いこと、
(2) IREE 3.11 の local-task device は入力 162 個以上の関数を呼ぶと落ちること
（nabla.iree は iree-too-many-inputs で拒否する。docs/iree-upstream-bugs.md の 6。
葉 309 個の grad の jit がこれで落ちた）から、葉の個数を抑える。上限の検査そのものは
tests/iree/input-limit-test.lisp が確かめる。"
  (let ((count 0))
    (nabla:tree-map (lambda (n)
                      (when (or (null max-leaves) (< (incf count) (1+ max-leaves)))
                        (make-random-array
                         (make-array-spec (ecase (mod n 3) (0 '()) (1 '(3)) (2 '(2 3))) dtype)
                         :seed (+ 1000 n))))
                    spec)))

(defparameter *pytree-leaf-op* (nabla:with-tracing (x) (tanh (+ x x)))
  "木の葉ごとに適用する、トレースできる演算。")

(defparameter *pytree-add-op* (nabla:with-tracing (a b) (+ a b)))

(defun pytree-map-function ()
  "木を受け取り、葉ごとに *PYTREE-LEAF-OP* を適用した同じ形の木を返す TRACEABLE-FUNCTION。"
  (nabla::%make-traceable-function
   (list (gensym "TREE"))
   (lambda (tree) (nabla:tree-map (lambda (x) (funcall *pytree-leaf-op* x)) tree))))

(defun pytree-loss-function ()
  "葉を1つ以上持つ木を受け取り、葉ごとの (reduce-sum (tanh (+ x x))) の総和（rank 0）を返す
TRACEABLE-FUNCTION。"
  (nabla::%make-traceable-function
   (list (gensym "TREE"))
   (lambda (tree)
     (reduce (lambda (acc x) (funcall *pytree-add-op* acc x))
             (mapcar (lambda (x) (nabla:reduce-sum (funcall *pytree-leaf-op* x)))
                     (nabla:tree-leaves tree))))))

(defun pytree-loss-oracle (treedef)
  "PYTREE-LOSS-FUNCTION と同じ関数を、TREEDEF の木の葉をばらした f64 配列を引数に取る普通の
関数にしたもの（CENTRAL-DIFFERENCE-GRADIENT に渡すオラクル。grad には依存しない）。"
  (let ((loss (pytree-loss-function)))
    (lambda (&rest leaves) (funcall loss (nabla:tree-unflatten treedef leaves)))))

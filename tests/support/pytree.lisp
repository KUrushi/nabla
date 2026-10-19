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

;;;; nb::var / nb::eqn / nb::graph / check-graph の性質（issue #29、u1a）。
;;;;
;;;; make-eqn / make-graph / check-graph / find-primitive は内部シンボル
;;;; （nb::）で呼ぶ。テストは内部を nb:: で使っている。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %count-definitions (graph)
  "GRAPH-INVARS・GRAPH-CONSTANTS の var・各 eqn の outvars に、それぞれの var
が何回現れるかを数える。check-graph とは別の実装で SSA を検算するための
オラクル。返り値は (var . count) の alist。"
  (let ((counts '()))
    (flet ((bump (v)
             (let ((entry (assoc v counts)))
               (if entry (incf (cdr entry)) (push (cons v 1) counts)))))
      (dolist (v (nb:graph-invars graph)) (bump v))
      (dolist (entry (nb:graph-constants graph)) (bump (car entry)))
      (dolist (eqn (nb:graph-eqns graph)) (dolist (v (nb:eqn-outvars eqn)) (bump v))))
    counts))

(defun %all-references-defined-p (graph)
  "各 eqn の invars と graph-outvars が、その時点までに定義済みの var だけを
参照していることを、check-graph とは別に検算する。"
  (let ((defined (make-hash-table :test 'eq)))
    (dolist (v (nb:graph-invars graph)) (setf (gethash v defined) t))
    (dolist (entry (nb:graph-constants graph)) (setf (gethash (car entry) defined) t))
    (dolist (eqn (nb:graph-eqns graph))
      (dolist (v (nb:eqn-invars eqn))
        (unless (gethash v defined) (return-from %all-references-defined-p nil)))
      (dolist (v (nb:eqn-outvars eqn)) (setf (gethash v defined) t)))
    (dolist (v (nb:graph-outvars graph))
      (unless (gethash v defined) (return-from %all-references-defined-p nil)))
    t))

(test ir/ssa-holds-for-random-recipes
  "ランダムなレシピから BUILD-GRAPH した graph に CHECK-GRAPH が通り、
さらに独立に数えた「各 var の定義回数 = 1、各参照は定義済み」も真になる。"
  (is (check-it (generator (graph-recipe))
                (lambda (recipe)
                  (let ((graph (build-graph recipe)))
                    (and (eq graph (nb::check-graph graph))
                         (every (lambda (entry) (= 1 (cdr entry))) (%count-definitions graph))
                         (= (length (%count-definitions graph)) (recipe-var-count recipe))
                         (%all-references-defined-p graph))))
                :regression-id ir/ssa-holds-for-random-recipes
                :regression-file (regression-path "ir-ssa-holds-for-random-recipes"))))

(test ir/make-eqn-out-aval-matches-abstract-eval
  "make-eqn が作る eqn の唯一の outvar の aval は、abstract-eval を直接
呼んだ結果と equalp で一致する。"
  (is (check-it (generator (array-spec))
                (lambda (spec)
                  (let* ((in-aval (nb:make-aval (array-spec-shape spec) (array-spec-dtype spec)))
                         (v (nb::make-var in-aval))
                         (eqn (nb::make-eqn :%test-neg (list v)))
                         (direct (funcall (nb::primitive-abstract-eval (nb::find-primitive :%test-neg))
                                          (list in-aval))))
                    (equalp direct (nb:var-aval (first (nb:eqn-outvars eqn))))))
                :regression-id ir/make-eqn-out-aval-matches-abstract-eval
                :regression-file (regression-path "ir-make-eqn-out-aval-matches-abstract-eval"))))

(test ir/make-eqn-invars-are-eq-to-arguments
  "eqn-invars の各要素は、make-eqn に渡した var と EQ。"
  (let* ((a (nb::make-var (nb:make-aval '(2 3) :f32)))
         (b (nb::make-var (nb:make-aval '(2 3) :f32)))
         (eqn (nb::make-eqn :%test-add (list a b))))
    (is (eq a (first (nb:eqn-invars eqn))))
    (is (eq b (second (nb:eqn-invars eqn))))))

(test ir/make-eqn-prim-is-eq-to-find-primitive
  "eqn-prim は (find-primitive name) と EQ。"
  (let* ((v (nb::make-var (nb:make-aval '(2 3) :f32)))
         (eqn (nb::make-eqn :%test-neg (list v))))
    (is (eq (nb::find-primitive :%test-neg) (nb:eqn-prim eqn)))))

(test ir/make-eqn-params-are-normalized-to-declared-order
  "eqn-params は宣言順の plist になる。呼び出し順を入れ替えて渡しても
同じ plist になる（%test-two-params は :a :b の宣言順に対して :b を先に
渡す）。"
  (let* ((v (nb::make-var (nb:make-aval '(6) :f32)))
         (eqn (nb::make-eqn :%test-reshape (list v) :shape '(2 3)))
         (v2 (nb::make-var (nb:make-aval '(2) :f32)))
         (eqn2 (nb::make-eqn :%test-two-params (list v2) :b 2 :a 1)))
    (is (equal '(:shape (2 3)) (nb:eqn-params eqn)))
    (is (equal '(:a 1 :b 2) (nb:eqn-params eqn2)))))

(test ir/check-graph-detects-use-before-def
  "eqn2 が eqn1 の出力に依存する2eqnのグラフで、eqn の順序を入れ替えて
use-before-def にすると MALFORMED-GRAPH。正しい順序では CHECK-GRAPH が通る
ことも確かめる。"
  (let* ((v (nb::make-var (nb:make-aval '(2 3) :f32)))
         (eqn1 (nb::make-eqn :%test-neg (list v)))
         (out1 (first (nb:eqn-outvars eqn1)))
         (eqn2 (nb::make-eqn :%test-neg (list out1)))
         (out2 (first (nb:eqn-outvars eqn2)))
         (ok (nb::make-graph (list v) (list eqn1 eqn2) (list out2)))
         (broken (nb::make-graph (list v) (list eqn2 eqn1) (list out2))))
    (is (eq ok (nb::check-graph ok)))
    (signals nb::malformed-graph (nb::check-graph broken))))

(test ir/check-graph-detects-duplicate-invar
  "invars に同じ var を2回入れると MALFORMED-GRAPH。"
  (let* ((v (nb::make-var (nb:make-aval '(2) :f32)))
         (graph (nb::make-graph (list v v) '() (list v))))
    (signals nb::malformed-graph (nb::check-graph graph))))

(test ir/check-graph-detects-undefined-outvar
  "outvars に未定義の var を入れると MALFORMED-GRAPH。"
  (let* ((v (nb::make-var (nb:make-aval '(2) :f32)))
         (undefined (nb::make-var (nb:make-aval '(2) :f32)))
         (graph (nb::make-graph (list v) '() (list undefined))))
    (signals nb::malformed-graph (nb::check-graph graph))))

(test ir/check-graph-detects-constant-aval-mismatch
  "constants の (var . array) の aval が var の aval と食い違うと
MALFORMED-GRAPH。"
  (let* ((array (make-random-array (make-array-spec '(2 3) :f32)))
         (v (nb::make-var (nb:make-aval '(3 2) :f32)))
         (graph (nb::make-graph '() '() (list v) (list (cons v array)))))
    (signals nb::malformed-graph (nb::check-graph graph))))

(test ir/print-object-does-not-error
  "var / graph の print-object が単に呼べる（デバッグ表示なので厳密な文字列
一致は求めない）。"
  (let* ((v (nb::make-var (nb:make-aval '(2 3) :f32)))
         (graph (nb::make-graph (list v) '() (list v))))
    (is (stringp (format nil "~A" v)))
    (is (stringp (format nil "~A" graph)))))

;;;; テスト専用プリミティブ（issue #29、u1a）。
;;;;
;;;; 名前は %TEST- 接頭辞にして、wave 2 が defprimitive する本物の
;;;; プリミティブ（add など）と衝突させない。整数・整数リスト・キーワード
;;;; の3種の params を網羅する（%TEST-RESHAPE = 整数リスト、%TEST-REDUCE =
;;;; 整数、%TEST-CONVERT = キーワード）。ABSTRACT-EVAL のみを持ち（EMIT /
;;;; EAGER は省略。u1a のスコープ外）、make-eqn / check-graph の配管を
;;;; 確かめるためだけに使う。

(in-package #:nabla.tests)

(nb:defprimitive %test-neg ()
  :abstract-eval (lambda (in-avals) (first in-avals)))

(nb:defprimitive %test-add ()
  :abstract-eval
  (lambda (in-avals)
    (destructuring-bind (a b) in-avals
      (unless (equal (nb:aval-shape a) (nb:aval-shape b))
        (error 'nb:primitive-error :name :%test-add :in-avals in-avals
               :format-control "shape が一致しない: ~S / ~S"
               :format-arguments (list (nb:aval-shape a) (nb:aval-shape b))))
      (unless (eq (nb:aval-dtype a) (nb:aval-dtype b))
        (error 'nb:primitive-error :name :%test-add :in-avals in-avals
               :format-control "dtype が一致しない: ~S / ~S"
               :format-arguments (list (nb:aval-dtype a) (nb:aval-dtype b))))
      a)))

(nb:defprimitive %test-reshape (:shape)
  :abstract-eval
  (lambda (in-avals &key shape)
    (let ((in (first in-avals)))
      (unless (= (nb:aval-size in) (reduce #'* shape :initial-value 1))
        (error 'nb:primitive-error :name :%test-reshape :in-avals in-avals
               :format-control "要素数が一致しない: ~S → ~S" :format-arguments (list (nb:aval-shape in) shape)))
      (nb:make-aval shape (nb:aval-dtype in)))))

(nb:defprimitive %test-convert (:dtype)
  :abstract-eval
  (lambda (in-avals &key dtype)
    (nb:make-aval (nb:aval-shape (first in-avals)) dtype)))

(nb:defprimitive %test-reduce (:axis)
  :abstract-eval
  (lambda (in-avals &key axis)
    (let* ((in (first in-avals))
           (shape (nb:aval-shape in)))
      (unless (< -1 axis (length shape))
        (error 'nb:primitive-error :name :%test-reduce :in-avals in-avals
               :format-control "axis ~S が shape ~S の範囲外" :format-arguments (list axis shape)))
      (nb:make-aval (append (subseq shape 0 axis) (subseq shape (1+ axis))) (nb:aval-dtype in)))))

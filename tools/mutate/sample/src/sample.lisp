;;;; sample.lisp -- runner を試すためのサンプル関数

(defpackage #:nabla.mutate.sample
  (:use #:cl)
  (:export #:clamp #:mean))

(in-package #:nabla.mutate.sample)

(defun clamp (x lo hi)
  "X を [LO, HI] に収める。"
  (cond ((< x lo) lo)
        ((> x hi) hi)
        (t x)))

(defun mean (numbers)
  "NUMBERS（空でないリスト）の算術平均を返す。"
  (/ (reduce #'+ numbers) (length numbers)))

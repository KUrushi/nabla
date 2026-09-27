;;;; weak.lisp -- わざと弱いテスト。エラーが出ないことしか確かめない。

(defpackage #:nabla.mutate.sample.weak-tests
  (:use #:cl)
  (:export #:run-tests))

(in-package #:nabla.mutate.sample.weak-tests)

(defun run-tests ()
  "境界値も戻り値も確かめない、わざと弱いテスト。
nabla-mutate の runner を確かめるための例なので、ここでは意図的に
甘くしている。"
  (and (numberp (nabla.mutate.sample:clamp 5 0 10))
       (numberp (nabla.mutate.sample:mean '(1 2 3)))))

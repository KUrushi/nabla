;;;; strong.lisp -- 値と境界を確かめる強いテスト

(defpackage #:nabla.mutate.sample.strong-tests
  (:use #:cl)
  (:export #:run-tests))

(in-package #:nabla.mutate.sample.strong-tests)

(defun run-tests ()
  "clamp / mean の戻り値と境界（下端・上端・範囲内）を確かめる。"
  (and (= (nabla.mutate.sample:clamp 5 0 10) 5)
       (= (nabla.mutate.sample:clamp -1 0 10) 0)
       (= (nabla.mutate.sample:clamp 11 0 10) 10)
       (= (nabla.mutate.sample:clamp 0 0 10) 0)
       (= (nabla.mutate.sample:clamp 10 0 10) 10)
       (= (nabla.mutate.sample:mean '(1 2 3)) 2)))

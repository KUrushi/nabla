;;;; defmodule のモジュールを scan の carry・ys にする（issue #201、small）。
;;;;
;;;; 守らせる性質: RNN のセルの状態 (h, c) をモジュールにして scan の init にすると、
;;;; - 結果の carry と ys は同じクラスのモジュールで（ys の各スロットは先頭の軸に積まれる）、
;;;;   値は Lisp のループで回した参照実装と一致する。
;;;; - init のモジュールについての grad は、同じクラスのモジュールで、スロットごとに
;;;;   f64 の中心差分と一致する。
;;;; - 本体が別のクラスのモジュールを返すと SCAN-CARRY-MISMATCH。

(in-package #:nabla.nn.tests)

(defmodule rc-state ((h :param) (c :param)))

(defmodule rc-other ((h :param) (c :param)))

(defparameter *rc-h* (nabla:with-tracing (h w x) (tanh (+ (* h w) x))))
(defparameter *rc-c* (nabla:with-tracing (c h) (+ c h)))

(defun %rc-cell (state x w)
  "state (h, c) と入力 x から、新しい状態と y (= 新しい状態) を返す。w は loop 不変の重み。"
  (let* ((h (funcall *rc-h* (rc-state-h state) w x))
         (new (make-rc-state :h h :c (funcall *rc-c* (rc-state-c state) h))))
    (values new new)))

(defun %rc-rand (shape seed)
  (make-random-array (make-array-spec shape :f64) :seed seed))

(defun %rc-body (w)
  (nabla::%make-traceable-function (list 'state 'x) (lambda (state x) (%rc-cell state x w))))

(defun %rc-row (array i)
  (let* ((row (make-array (rest (array-dimensions array)) :element-type (array-element-type array)))
         (size (array-total-size row)))
    (dotimes (j size row)
      (setf (row-major-aref row j) (row-major-aref array (+ (* i size) j))))))

(defun %rc-stack (rows)
  (let* ((size (array-total-size (first rows)))
         (out (make-array (cons (length rows) (array-dimensions (first rows)))
                          :element-type (array-element-type (first rows)))))
    (loop for row in rows for i from 0
          do (dotimes (j size) (setf (row-major-aref out (+ (* i size) j)) (row-major-aref row j))))
    out))

(defun %rc-reference (init xs w)
  "Lisp のループによる scan。(values 最終の状態 ys のモジュール)。"
  (let ((state init) (ys '()))
    (dotimes (i (array-dimension xs 0))
      (multiple-value-bind (new y) (%rc-cell state (%rc-row xs i) w)
        (setf state new)
        (push y ys)))
    (setf ys (nreverse ys))
    (values state
            (make-rc-state :h (%rc-stack (mapcar #'rc-state-h ys))
                           :c (%rc-stack (mapcar #'rc-state-c ys))))))

(defun %rc-close (a b)
  (and (typep a 'rc-state) (typep b 'rc-state)
       (allclose (rc-state-h a) (rc-state-h b) :dtype :f64)
       (allclose (rc-state-c a) (rc-state-c b) :dtype :f64)))

(test (module-control/scan-over-a-module-carry-matches-reference :suite :nabla.small)
  "モジュールの carry の scan は、carry も ys も同じクラスのモジュールで、参照実装と一致する。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 4)))
                (lambda (case)
                  (destructuring-bind (seed length) case
                    (let ((init (make-rc-state :h (%rc-rand '(3) seed) :c (%rc-rand '(3) (+ seed 1))))
                          (xs (%rc-rand (list length 3) (+ seed 2)))
                          (w (%rc-rand '(3) (+ seed 3))))
                      (multiple-value-bind (state ys) (nabla:scan (%rc-body w) init xs)
                        (multiple-value-bind (expected-state expected-ys) (%rc-reference init xs w)
                          (and (%rc-close state expected-state)
                               (%rc-close ys expected-ys)
                               (equal (list length 3) (array-dimensions (rc-state-h ys)))))))))
                :regression-id module-control/scan-over-a-module-carry-matches-reference
                :regression-file (regression-path "module-control-scan-matches-reference"
                                                  :package "NABLA.NN.TESTS"))))

(test (module-control/grad-through-a-module-carry-scan-matches-central-difference :suite :nabla.small)
  "init のモジュールについての grad は同じクラスのモジュールで、スロットごとに f64 の中心差分と一致する。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 3)))
                (lambda (case)
                  (destructuring-bind (seed length) case
                    (let* ((xs (%rc-rand (list length 3) (+ seed 2)))
                           (w (%rc-rand '(3) (+ seed 3)))
                           (init (make-rc-state :h (%rc-rand '(3) seed) :c (%rc-rand '(3) (+ seed 1))))
                           (loss (nabla::%make-traceable-function
                                  (list 'init)
                                  (lambda (init)
                                    (multiple-value-bind (state ys) (nabla:scan (%rc-body w) init xs)
                                      (funcall (nabla:with-tracing (a b) (+ a b))
                                               (nabla:reduce-sum (rc-state-c state))
                                               (nabla:reduce-sum (rc-state-h ys)))))))
                           (g (funcall (nabla:grad loss) init))
                           (expected (central-difference-gradient
                                      (lambda (h c) (funcall loss (make-rc-state :h h :c c)))
                                      (list (rc-state-h init) (rc-state-c init)))))
                      (and (typep g 'rc-state)
                           (allclose (rc-state-h g) (first expected) :dtype :f64
                                                                      :rtol *autodiff-rtol* :atol *autodiff-atol*)
                           (allclose (rc-state-c g) (second expected) :dtype :f64
                                                                      :rtol *autodiff-rtol* :atol *autodiff-atol*)))))
                :regression-id module-control/grad-through-a-module-carry-scan
                :regression-file (regression-path "module-control-grad-matches-fd"
                                                  :package "NABLA.NN.TESTS"))))

(test (module-control/scan-rejects-a-body-that-returns-another-module-class :suite :nabla.small)
  "本体が init と別のクラスのモジュールを返すと（スロットの葉が同じでも）SCAN-CARRY-MISMATCH。"
  (let ((init (make-rc-state :h (%rc-rand '(3) 1) :c (%rc-rand '(3) 2))))
    (signals nabla:scan-carry-mismatch
      (nabla:scan (nabla::%make-traceable-function
                   (list 'state 'x)
                   (lambda (state x)
                     (declare (ignore x))
                     (values (make-rc-other :h (rc-state-h state) :c (rc-state-c state)) '())))
                  init '() :length 2))))

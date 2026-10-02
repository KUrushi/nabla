;;;; primitives/stop-gradient: stop-gradient プリミティブ（issue #80）。
;;;;
;;;; 値は入力そのままの恒等で、jvp（接線）は常に symbolic zero（ルールは
;;;; src/ad/rules-elementwise.lisp）。JAX の lax.stop_gradient に相当する。
;;;; 任意の dtype（:I1 を含む）・shape を通す。
;;;;
;;;; StableHLO には恒等の op が無いので、stablehlo.optimization_barrier を
;;;; 1オペランドで出す（docs/stablehlo-ops.md）。値は変えず、実行系が
;;;; コンパイルできることは実行系側のフィクスチャで確かめてある（tests/fixtures/stablehlo/ops/）。

(in-package #:nabla)

(defun %stop-gradient-abstract-eval (in-avals)
  (%check-arity :stop-gradient in-avals 1)
  (first in-avals))

(defun %stop-gradient-emit (in-names in-avals out-name)
  "\"<out-name> = stablehlo.optimization_barrier <a> : <T>\"。"
  (format nil "~A = stablehlo.optimization_barrier ~A : ~A"
          out-name (first in-names) (tensor-type-string (first in-avals))))

(defun %stop-gradient-eager (arrays)
  "入力の中身が等しい新しい配列（元の配列とは別物）を返す。"
  (let* ((array (first arrays))
         (result (make-array (array-dimensions array) :element-type (array-element-type array))))
    (dotimes (i (array-total-size result) result)
      (setf (row-major-aref result i) (row-major-aref array i)))))

(defprimitive stop-gradient ()
  :abstract-eval (lambda (in-avals) (%stop-gradient-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore out-aval))
          (%stop-gradient-emit in-names in-avals out-name))
  :eager (lambda (arrays in-avals)
           (declare (ignore in-avals))
           (%stop-gradient-eager arrays)))

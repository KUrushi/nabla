;;;; vjp 変換した graph の IREE 経由 end-to-end テスト（issue #82）。
;;;;
;;;; nb::vjp-graph が作った graph（主値 ++ 出力の余接線 → 主値の出力 ++ 入力の余接線）
;;;; を emit-stablehlo → backend-compile/load/invoke した結果が、同じ graph の
;;;; eval-graph と一致する。実プリミティブの transpose ルールは #83 で揃うので、
;;;; ここでは実プリミティブの emit / eager に委ねる線形テスト専用プリミティブ
;;;; （%IREE-VJP-NEG / %IREE-VJP-MUL）に jvp / transpose ルールを付けて使う
;;;; （nabla/iree/tests は nabla/tests に依存しないので、tests/test-primitives.lisp の
;;;; %test-* は使えない）。

(in-package #:nabla.iree.tests)

(defun %delegate-emit (name)
  (let ((primitive (nb::find-primitive name)))
    (lambda (in-names in-avals out-name out-aval)
      (funcall (nb::primitive-emit primitive) in-names in-avals out-name out-aval))))

(defun %delegate-eager (name)
  (let ((primitive (nb::find-primitive name)))
    (lambda (arrays in-avals)
      (funcall (nb::primitive-eager primitive) arrays in-avals))))

(nb:defprimitive %iree-vjp-neg ()
  :abstract-eval (lambda (in-avals) (first in-avals))
  :emit (%delegate-emit :neg)
  :eager (%delegate-eager :neg))

(nb:defprimitive %iree-vjp-mul ()
  :abstract-eval (lambda (in-avals) (first in-avals))
  :emit (%delegate-emit :mul)
  :eager (%delegate-eager :mul))

(nb::def-jvp-rule %iree-vjp-neg (primals out tangents)
  (declare (ignore primals out))
  (nb::%trace-eqn :%iree-vjp-neg (list (first tangents))))

(nb::def-transpose-rule %iree-vjp-neg (ct invars)
  (declare (ignore invars))
  (list (nb::%trace-eqn :%iree-vjp-neg (list ct))))

(nb::def-jvp-rule %iree-vjp-mul (primals out tangents)
  (declare (ignore out))
  (destructuring-bind (a b) primals
    (destructuring-bind (ta tb) tangents
      (nb::add-tangents
       (if (nb::symbolic-zero-p ta) ta (nb::%trace-eqn :%iree-vjp-mul (list ta b)))
       (if (nb::symbolic-zero-p tb) tb (nb::%trace-eqn :%iree-vjp-mul (list a tb)))))))

(nb::def-transpose-rule %iree-vjp-mul (ct invars)
  (destructuring-bind (a b) invars
    (cond ((and (nb::undefined-primal-p a) (not (nb::undefined-primal-p b)))
           (list (nb::%trace-eqn :%iree-vjp-mul (list ct b)) nil))
          ((and (nb::undefined-primal-p b) (not (nb::undefined-primal-p a)))
           (list nil (nb::%trace-eqn :%iree-vjp-mul (list ct a))))
          (t (error 'nb:autodiff-error :format-control "%iree-vjp-mul は片側だけが線形のときだけ転置できる")))))

;; 接線の足し算は実プリミティブの add になる。その transpose ルールは #83 が書く。
;; それまで、無ければこのテストの中でだけ補う。
(unless (nb::primitive-transpose (nb::find-primitive :add))
  (nb::def-transpose-rule add (ct invars)
    (declare (ignore invars))
    (list ct ct)))

(defun %vjp-iree-matches-eval-graph-p (backend graph arrays)
  "GRAPH（f32）を ARRAYS で IREE 実行した全出力が eval-graph の結果と一致するか。"
  (let* ((module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph))))
         (device-arrays '())
         (results '()))
    (unwind-protect
         (progn
           (setf device-arrays (mapcar (lambda (a) (to-device a backend :dtype :f32)) arrays))
           (setf results (multiple-value-list (apply #'nabla:backend-invoke backend module "main" device-arrays)))
           (let ((expected (multiple-value-list (apply #'nb:eval-graph graph arrays))))
             (and (= (length results) (length expected))
                  (every (lambda (r e) (allclose (to-host r) e :dtype :f32)) results expected))))
      (dolist (r results) (release-device-array r))
      (dolist (da device-arrays) (release-device-array da))
      (nabla:backend-unload backend module))))

(define-iree-test vjp/iree-matches-eval-graph
    "(lambda (x y) (values (- (* x y)) (+ x y)))（2出力）を graph にして vjp-graph（全入力と、
x だけの2通り）→ emit-stablehlo → IREE で実行した結果は、同じ graph の eval-graph と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 4))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((shape (loop for k from 1 to (mod seed 3) collect (1+ (mod (+ seed k) 3))))
                           (aval (nb:make-aval shape :f32))
                           (spec (make-array-spec shape :f32))
                           (graph (nb::%call-with-fresh-trace
                                   (list aval aval)
                                   (lambda (x y)
                                     (let ((product (nb::%trace-eqn :%iree-vjp-mul (list x y))))
                                       (values (nb::%trace-eqn :%iree-vjp-neg (list product))
                                               (nb::%t-add x y))))))
                           ;; 主値 x, y と、2つの出力の余接線。
                           (arrays (loop for i below 4 collect (make-random-array spec :seed (+ seed i)))))
                      (and (%vjp-iree-matches-eval-graph-p backend (nb::vjp-graph graph) arrays)
                           (%vjp-iree-matches-eval-graph-p backend (nb::vjp-graph graph :nonzero '(t nil))
                                                           arrays))))
                  :regression-id vjp/iree-matches-eval-graph
                  :regression-file (regression-path "iree-vjp-matches-eval-graph" :package "NABLA.IREE.TESTS")))))

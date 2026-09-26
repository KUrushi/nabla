;;;; mutate-tests.lisp -- mutate-form の性質

(in-package #:nabla.mutate.tests)

(in-suite :nabla-mutate)

(defparameter *arith-ops* '(+ - * /))
(defparameter *boundary-ops* '(< <= > >=))

(defun %count-diffs (a b)
  "同じ形をしている（はずの）A と B のリーフの違いを数える。"
  (cond
    ((and (consp a) (consp b))
     (+ (%count-diffs (car a) (car b)) (%count-diffs (cdr a) (cdr b))))
    ((equal a b) 0)
    (t 1)))

(test mutate-form-arith-swap-changes-exactly-one-node-and-is-involutive
  (is (check-it
       (generator (tuple (integer 0 3) (integer -20 20) (integer -20 20)))
       (lambda (input)
         (destructuring-bind (op-index a b) input
           (let ((form (list (nth op-index *arith-ops*) a b)))
             (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form form :arith-swap)
               (and applied
                    (not (equal mutated form))
                    (= 1 (%count-diffs form mutated))
                    (equal form (nabla.mutate:mutate-form mutated :arith-swap))))))))))

(test mutate-form-boundary-changes-exactly-one-node-and-is-involutive
  (is (check-it
       (generator (tuple (integer 0 3) (integer -20 20) (integer -20 20)))
       (lambda (input)
         (destructuring-bind (op-index a b) input
           (let ((form (list (nth op-index *boundary-ops*) a b)))
             (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form form :boundary)
               (and applied
                    (not (equal mutated form))
                    (= 1 (%count-diffs form mutated))
                    (equal form (nabla.mutate:mutate-form mutated :boundary))))))))))

(test mutate-form-branch-swap-changes-exactly-one-node-and-is-involutive
  (is (check-it
       (generator (tuple (integer -10 10) (integer -10 10) (integer 0 20)))
       (lambda (input)
         (destructuring-bind (c a width) input
           (let* ((b (+ a 1 width))
                  (form (list 'if (list '> c 0) a b)))
             (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form form :branch-swap)
               (and applied
                    (not (equal mutated form))
                    (equal form (nabla.mutate:mutate-form mutated :branch-swap))))))))))

(test mutate-form-constant-never-identity
  (is (check-it
       (generator (integer -50 50))
       (lambda (n)
         (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form n :constant)
           (and applied (not (equal mutated n))))))))

(test mutate-form-skips-arid-nodes
  "arid node（ここでは format）の中の算術は変異させない。"
  (multiple-value-bind (mutated applied)
      (nabla.mutate:mutate-form '(format t "~A" (+ 1 2)) :arith-swap)
    (is-false applied)
    (is (equal mutated '(format t "~A" (+ 1 2))))))

(test mutate-form-no-match-returns-unapplied
  (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form '(list 1 2 3) :branch-swap)
    (is-false applied)
    (is (equal mutated '(list 1 2 3)))))

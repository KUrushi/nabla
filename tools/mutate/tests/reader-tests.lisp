;;;; reader-tests.lisp -- read-source-forms / definitions-in-range / arid-node-p

(in-package #:nabla.mutate.tests)

(in-suite :nabla-mutate)

(defun %make-def (index start end)
  (nabla.mutate:make-source-form
   :form (list 'defun (intern (format nil "F~D" index)) nil 1)
   :start-line start
   :end-line end))

(test definitions-in-range-matches-manual-filter
  "definitions-in-range は、行範囲が問い合わせと重なる定義だけを、
元の順序のまま返す。"
  (is (check-it
       (generator (tuple (list (tuple (integer 1 40) (integer 0 10)))
                          (integer 1 40)
                          (integer 0 20)))
       (lambda (input)
         (destructuring-bind (specs query-start query-width) input
           (let* ((defs (loop for (start width) in specs
                               for i from 0
                               collect (%make-def i start (+ start width))))
                  (query-end (+ query-start query-width))
                  (expected (remove-if-not
                             (lambda (d)
                               (and (<= (nabla.mutate:source-form-start-line d) query-end)
                                    (>= (nabla.mutate:source-form-end-line d) query-start)))
                             defs))
                  (actual (nabla.mutate:definitions-in-range defs query-start query-end)))
             (equal (mapcar #'nabla.mutate:source-form-form expected)
                    (mapcar #'nabla.mutate:source-form-form actual))))))))

(test definitions-in-range-excludes-non-definitions
  "defun / defmethod / defmacro / defprimitive 以外のトップレベルフォームは、
行範囲が重なっていても対象にならない。"
  (let* ((not-a-def (nabla.mutate:make-source-form :form '(defvar *x* 1)
                                                     :start-line 1 :end-line 1))
         (a-def (%make-def 0 1 1)))
    (is (equal (list (nabla.mutate:source-form-form a-def))
               (mapcar #'nabla.mutate:source-form-form
                       (nabla.mutate:definitions-in-range (list not-a-def a-def) 1 1))))))

(test arid-node-p-detects-known-heads
  "format / declare など、変異させても意味のないフォームを arid とみなす。"
  (is-true (nabla.mutate:arid-node-p '(format t "~A" x)))
  (is-true (nabla.mutate:arid-node-p '(declare (type fixnum x))))
  (is-true (nabla.mutate:arid-node-p '(error "bad ~A" x)))
  (is-false (nabla.mutate:arid-node-p '(+ 1 2)))
  (is-false (nabla.mutate:arid-node-p 42)))

(test read-source-forms-line-numbers-skip-blank-lines-between-forms
  "フォームの間に空行があっても、start-line はその定義の最初の非空白
行を指す（直前のフォームの末尾で消費された改行1文字を、次のフォームの
開始位置に含めない）。"
  (let ((tmp (uiop:with-temporary-file (:pathname p :type "lisp" :keep t) p)))
    (unwind-protect
         (progn
           (with-open-file (s tmp :direction :output :if-exists :supersede)
             (write-string
              (format nil "(defun clamp (x lo hi)~%  (max lo (min hi x)))~%~%~%(defun mean (xs)~%  (/ (reduce #'+ xs) (length xs)))~%")
              s))
           (let ((forms (nabla.mutate:read-source-forms tmp)))
             (is (= 2 (length forms)))
             (destructuring-bind (clamp-def mean-def) forms
               (is (= 1 (nabla.mutate:source-form-start-line clamp-def)))
               (is (= 2 (nabla.mutate:source-form-end-line clamp-def)))
               (is (= 5 (nabla.mutate:source-form-start-line mean-def)))
               (is (= 6 (nabla.mutate:source-form-end-line mean-def))))))
      (ignore-errors (delete-file tmp)))))

(test read-source-forms-line-numbers-skip-comments-above-forms
  "定義の直前にあるコメント行やブロックコメントは、その定義の start-line
に含めない（コメントだけを触った変更が、下の定義を変異対象に選んでしまう
のを防ぐ）。"
  (let ((tmp (uiop:with-temporary-file (:pathname p :type "lisp" :keep t) p)))
    (unwind-protect
         (progn
           (with-open-file (s tmp :direction :output :if-exists :supersede)
             (write-string
              (format nil ";;;; header~%~%;; helper~%(defun f (x) x) ; trailing~%~%#| block~%comment |#~%(defun g (x) x)~%")
              s))
           (let ((forms (nabla.mutate:read-source-forms tmp)))
             (is (= 2 (length forms)))
             (destructuring-bind (f-def g-def) forms
               (is (= 4 (nabla.mutate:source-form-start-line f-def)))
               (is (= 8 (nabla.mutate:source-form-start-line g-def))))))
      (ignore-errors (delete-file tmp)))))

(test read-source-forms-tracks-in-package
  "in-package の後のフォームは、そのパッケージで読まれる（シンボルの
パッケージで確認する）。"
  (let ((custom (or (find-package "NABLA-MUTATE-TEST-PKG")
                     (make-package "NABLA-MUTATE-TEST-PKG" :use '("CL"))))
        (tmp (uiop:with-temporary-file (:pathname p :type "lisp" :keep t) p)))
    (unwind-protect
         (progn
           (with-open-file (s tmp :direction :output :if-exists :supersede)
             (write-string
              (format nil "(in-package #:nabla-mutate-test-pkg)~%(defun sample-fn (x) (+ x 1))~%")
              s))
           (let ((forms (nabla.mutate:read-source-forms tmp)))
             (is (= 2 (length forms)))
             (let ((def (second forms)))
               (is (eq custom (nabla.mutate:source-form-package def)))
               (is (= 2 (nabla.mutate:source-form-start-line def))))))
      (ignore-errors (delete-file tmp))
      (ignore-errors (delete-package custom)))))

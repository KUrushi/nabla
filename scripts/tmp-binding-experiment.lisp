;;;; TEMP（#264 の実験）。消す。
(require :asdf)
(asdf:load-system "nabla/iree")
(asdf:load-system "nabla/nn")
(pushnew :nabla-no-demo *features*)
(defpackage #:tmp-exp (:use #:cl))
(in-package #:tmp-exp)
(setf nabla:*compile-cache-directory* nil)
(load (asdf:system-relative-pathname "nabla" "examples/transformer.lisp"))

(defclass capture-backend (nabla:backend) ())
(defmethod nabla:backend-target ((b capture-backend)) :local)
(defmethod nabla:backend-fingerprint ((b capture-backend)) (list "capture"))
(defmethod nabla:backend-compile ((b capture-backend) text) (throw 'captured text))

(defun sym (name) (find-symbol name (find-package "NABLA-EXAMPLE-TRANSFORMER")))

(defun transformer-args ()
  (let* ((optimizer (funcall (sym "MAKE-OPTIMIZER")))
         (model (funcall (sym "MAKE-EXAMPLE-TRANSFORMER") :seed 0))
         (state (nabla.nn:optimizer-init optimizer model)))
    (multiple-value-bind (tokens targets)
        (funcall (sym "SEQUENCES->BATCH") (funcall (sym "MAKE-COPY-SEQUENCES") 16 0))
      (values optimizer model state tokens targets))))

(let ((out (uiop:getenv "TMP_EXP_TEXT_OUT")))
  (when out
    (multiple-value-bind (optimizer model state tokens targets) (transformer-args)
      (let* ((step (funcall (sym "MAKE-TRAIN-STEP") optimizer :backend (make-instance 'capture-backend)))
             (text (catch 'captured (funcall step model state tokens targets) nil)))
        (with-open-file (s out :direction :output :if-exists :supersede) (write-string text s))
        (format t "~&EXP captured transformer step text ~D chars~%" (length text))))))

(defun now () (/ (get-internal-real-time) (/ internal-time-units-per-second 1000.0)))

(multiple-value-bind (optimizer model state tokens targets) (transformer-args)
  (let* ((step (funcall (sym "MAKE-TRAIN-STEP") optimizer))
         (t0 (now)))
    (multiple-value-bind (l m s) (funcall step model state tokens targets)

      (format t "~&EXP transformer first-call ~,1F ms flags=~A~%" (- (now) t0) (uiop:getenv "NABLA_TMP_IREE_EXTRA_FLAGS"))
      (let ((times '()))
        (dotimes (k 40)
          (let ((t1 (now)))
            (multiple-value-setq (l m s) (funcall step m s tokens targets))
            (push (- (now) t1) times)))
        (setf times (sort (subseq times 0 30) #'<))
        (format t "~&EXP transformer step median ~,3F ms min ~,3F ms flags=~A~%"
                (nth 15 times) (first times) (uiop:getenv "NABLA_TMP_IREE_EXTRA_FLAGS"))))))

;; nabla の葉 N 個の (reduce #'add) を jit
(defparameter *add* (nabla:with-tracing (a b) (+ a b)))
(defun chain-fn ()
  (nabla::%make-traceable-function
   (list (gensym))
   (lambda (xs) (reduce (lambda (a b) (funcall *add* a b)) xs))))
(dolist (n '(31 32 40 100))
  (let* ((xs (loop for i below n collect (let ((a (make-array 3 :element-type 'single-float)))
                                           (fill a (float i 1.0)) a)))
         (f (chain-fn))
         (jf (nabla:jit f :backend (nabla:make-backend :iree :target :local))))
    (handler-case
        (let ((r (funcall jf xs)))
          (format t "~&EXP nabla-chain n=~D OK ~A~%" n r))
      (error (c)
        (format t "~&EXP nabla-chain n=~D ERROR type=~S cause=~S msg=~A~%" n (type-of c)
                (and (typep c 'nabla:jit-compile-error) (type-of (nabla:jit-compile-error-condition c)))
                (subseq (remove #\Newline (princ-to-string c)) 0 (min 300 (length (princ-to-string c)))))))))

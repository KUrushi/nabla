;;;; cond* の medium テスト（issue #130）。
;;;;
;;;; cond* を含む graph を emit-stablehlo し、IREE の local backend でコンパイル・
;;;; 実行した結果が、eager（eval-graph）の結果と一致することを確かめる。
;;;; pred の両方の値で確かめる。

(in-package #:nabla.iree.tests)

(defun %cond-iree-matches-eager-p (backend module graph seed)
  (let* ((avals (mapcar #'nb:var-aval (nb:graph-invars graph)))
         (arrays (cons (make-array '() :element-type 'bit :initial-element (mod seed 2))
                       (loop for aval in (rest avals) for i from 0
                             collect (make-random-array
                                      (make-array-spec (nb:aval-shape aval) (nb:aval-dtype aval))
                                      :seed (+ seed i)))))
         (device-arrays nil)
         (results nil))
    (unwind-protect
         (progn
           (setf device-arrays
                 (loop for a in arrays for aval in avals
                       collect (to-device a backend :dtype (nb:aval-dtype aval))))
           (setf results (multiple-value-list
                          (apply #'nabla:backend-invoke backend module "main" device-arrays)))
           (let ((expected (multiple-value-list (apply #'nb:eval-graph graph arrays))))
             (and (= (length results) (length expected))
                  (every (lambda (r e) (allclose (to-host r) e :dtype :f32)) results expected))))
      (dolist (r results) (release-device-array r))
      (dolist (da device-arrays) (release-device-array da)))))

(defmacro %with-cond-iree-check ((backend graph name) message)
  `(let ((module (nabla:backend-load ,backend (nabla:backend-compile ,backend (nb:emit-stablehlo ,graph)))))
     (unwind-protect
          (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                        (lambda (seed) (%cond-iree-matches-eager-p ,backend module ,graph seed))
                        :regression-id ,(intern (string-upcase (format nil "cond/iree-~A" name)))
                        :regression-file (regression-path ,(format nil "iree-cond-~A" name)
                                                          :package "NABLA.IREE.TESTS"))
              ,message)
       (nabla:backend-unload ,backend module))))

(defparameter *cond-iree-avals*
  (list (nb:make-aval '() :i1) (nb:make-aval '(3 5) :f32) (nb:make-aval '(3 5) :f32)))

(define-iree-test cond/iree-multiple-outputs-match-eager
    "複数出力の cond* を IREE で実行した結果は、eager の結果と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (graph (nb::trace-to-graph
                (nb:with-tracing (p x y)
                  (multiple-value-bind (a b)
                      (nb:cond* p
                                (nb:with-tracing (u v) (values (+ u v) (* u v)))
                                (nb:with-tracing (u v) (values (- u v) u))
                                x y)
                    (values a (+ b 1))))
                *cond-iree-avals*)))
    (%with-cond-iree-check (backend graph "multiple-outputs")
      "IREE の cond* の結果が eager と一致しなかった")))

(define-iree-test cond/iree-closure-and-nested-match-eager
    "閉包で捕まえた外側の値と、入れ子の cond* も IREE の結果が eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (graph (nb::trace-to-graph
                (nb:with-tracing (p x y)
                  (nb:cond* p
                            (nb:with-tracing (u)
                              (nb:cond* (> (nb:reduce-sum u :axes '(0 1)) 0)
                                        (nb:with-tracing (w) (+ w y))
                                        (nb:with-tracing (w) (- w))
                                        u))
                            (nb:with-tracing (u) (* u y))
                            x))
                *cond-iree-avals*)))
    (%with-cond-iree-check (backend graph "closure-nested")
      "IREE の cond* の結果が eager と一致しなかった")))

(define-iree-test cond/iree-integer-operand-matches-eager
    "整数（:i32）の配列を operand に渡した cond* も IREE で動き、eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (graph (nb::trace-to-graph
                (nb:with-tracing (p x y)
                  (nb:cond* p
                            (nb:with-tracing (u k v) (+ u v))
                            (nb:with-tracing (u k v) (- u v))
                            x (make-array 2 :element-type '(signed-byte 32) :initial-element 7) y))
                *cond-iree-avals*)))
    (%with-cond-iree-check (backend graph "integer-operand")
      "IREE の cond* の結果が eager と一致しなかった")))

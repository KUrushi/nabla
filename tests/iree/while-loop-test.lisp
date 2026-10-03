;;;; while-loop プリミティブの medium テスト（issue #131）。
;;;;
;;;; stablehlo.while を IREE の local backend でコンパイル・実行した結果が、
;;;; eager（eval-graph）の結果と一致することを確かめる。limit は 0〜6（0回で
;;;; 終わる場合を含む）。graph のコンパイルは1回で、PBT の各試行は module を再利用する。

(in-package #:nabla.iree.tests)

(defun %wl-iree-graph ()
  "limit（cond が閉包で捕まえる）・step（body が閉包で捕まえる）・x を引数に取る while-loop。"
  (nb::trace-to-graph
   (nb:with-tracing (limit step x)
     (let ((result (nb:while-loop
                    (nb:with-tracing (c) (< (first c) limit))
                    (nb:with-tracing (c) (list (+ (first c) 1.0) (+ (* (second c) 0.5) step)))
                    (list (nb::%scalar-array 0.0 :f32) x))))
       (values (first result) (second result))))
   (list (nb:make-aval '() :f32) (nb:make-aval '(3 5) :f32) (nb:make-aval '(3 5) :f32))))

(defun %wl-iree-matches-eager-p (backend module graph seed)
  (let* ((limit (nb::%scalar-array (float (mod seed 7) 1.0) :f32))
         (step (make-random-array (make-array-spec '(3 5) :f32) :seed seed))
         (x (make-random-array (make-array-spec '(3 5) :f32) :seed (1+ seed)))
         (arrays (list limit step x))
         (device-arrays nil)
         (results nil))
    (unwind-protect
         (progn
           (setf device-arrays (mapcar (lambda (a) (to-device a backend :dtype :f32)) arrays))
           (setf results (multiple-value-list
                          (apply #'nabla:backend-invoke backend module "main" device-arrays)))
           (let ((expected (multiple-value-list (apply #'nb:eval-graph graph arrays))))
             (and (= (length results) (length expected))
                  (every (lambda (r e) (allclose (to-host r) e :dtype :f32)) results expected))))
      (dolist (r results) (release-device-array r))
      (dolist (da device-arrays) (release-device-array da)))))

(define-iree-test while-loop/iree-matches-eager
    "stablehlo.while を IREE でコンパイル・実行した結果は、eager の while-loop と一致する
（0回で終わる場合、閉包で捕まえた値を含む）。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (graph (%wl-iree-graph))
         (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
    (unwind-protect
         (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                       (lambda (seed) (%wl-iree-matches-eager-p backend module graph seed))
                       :regression-id while-loop/iree-matches-eager
                       :regression-file (regression-path "iree-while-loop-matches-eager"
                                                         :package "NABLA.IREE.TESTS"))
             "IREE の実行結果が eager の while-loop と一致しなかった")
      (nabla:backend-unload backend module))))

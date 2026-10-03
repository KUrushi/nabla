;;;; while-loop の jvp 変換した graph の medium テスト（issue #134）。
;;;;
;;;; jvp-graph した graph（carry に接線を足した stablehlo.while）を IREE の local backend で
;;;; コンパイル・実行した結果が、eager（eval-graph）の結果と一致することを確かめる。
;;;; y の初期値は定数で、接線はゼロから始まり本体で非ゼロになる（不動点の経路）。

(in-package #:nabla.iree.tests)

(defun %wlj-iree-graph ()
  "limit（cond が捕まえる）・w（body が捕まえる）・x を引数に取る f32 の while-loop の jvp。"
  (nb::jvp-graph
   (nb::trace-to-graph
    (nb:with-tracing (limit w x)
      (let ((result (nb:while-loop
                     (nb:with-tracing (c) (< (first c) limit))
                     (nb:with-tracing (c)
                       (list (+ (first c) 1.0)
                             (+ (* (second c) 0.5) (* w 0.1))
                             (+ (* (third c) 0.9) (nb:reduce-sum (* (second c) w) :axes '(0)))))
                     (list (nb::%scalar-array 0.0 :f32) x (nb::%scalar-array 0.0 :f32)))))
        (values (third result) (second result))))
    (list (nb:make-aval '() :f32) (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32)))
   ;; limit の接線は無し、w と x の接線あり
   :nonzero '(nil t t)))

(defun %wlj-iree-matches-eager-p (backend module graph seed)
  (let* ((arrays (list (nb::%scalar-array (float (mod seed 6) 1.0) :f32)
                       (make-random-array (make-array-spec '(3) :f32) :seed seed)
                       (make-random-array (make-array-spec '(3) :f32) :seed (+ seed 1))
                       (make-random-array (make-array-spec '(3) :f32) :seed (+ seed 2))
                       (make-random-array (make-array-spec '(3) :f32) :seed (+ seed 3))))
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

(define-iree-test while-loop-jvp/iree-matches-eager
    "while-loop を jvp 変換した graph を IREE でコンパイル・実行した結果は eager と一致する。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (graph (%wlj-iree-graph))
         (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
    (unwind-protect
         (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                       (lambda (seed) (%wlj-iree-matches-eager-p backend module graph seed))
                       :regression-id while-loop-jvp/iree-matches-eager
                       :regression-file (regression-path "iree-while-loop-jvp-matches-eager"
                                                         :package "NABLA.IREE.TESTS"))
             "jvp 変換した while-loop の IREE の実行結果が eager と一致しなかった")
      (nabla:backend-unload backend module))))

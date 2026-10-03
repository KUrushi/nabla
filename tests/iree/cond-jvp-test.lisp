;;;; cond の jvp / vjp（grad）変換した graph の medium テスト（issue #134）。
;;;;
;;;; jvp-graph と vjp-graph で変換した graph（主値の cond と線形な cond、転置した cond を含む）を
;;;; IREE の local backend でコンパイル・実行した結果が、eager（eval-graph）の結果と一致する
;;;; ことを確かめる。pred は sum(x) > 0 で、入力の乱数により両方の枝が選ばれる。

(in-package #:nabla.iree.tests)

(defun %cj-iree-primal-graph ()
  (nb::trace-to-graph
   (nb:with-tracing (x w v)
     (nb:cond* (> (nb:reduce-sum x :axes '(0)) 0.0)
               (nb:with-tracing (a b) (+ (* a b) (* a v)))
               (nb:with-tracing (a b) (* (* a a) w))
               x w))
   (list (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32))))

(defun %cj-iree-matches-eager-p (backend module graph seed n-inputs)
  (let* ((arrays (loop for i below n-inputs
                       collect (make-random-array (make-array-spec '(3) :f32) :seed (+ seed i))))
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

(defun %cj-iree-call-with-module (graph fn)
  "GRAPH をコンパイル・ロードして、(FN backend module) を呼ぶ。終わったらアンロードする。"
  (let* ((backend (nabla:find-backend :iree))
         (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
    (unwind-protect (funcall fn backend module)
      (nabla:backend-unload backend module))))

(define-iree-test cond-jvp/iree-jvp-graph-matches-eager
    "cond を jvp 変換した graph（主値の stablehlo.if と線形な stablehlo.if）は、IREE で
コンパイル・実行した結果が eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::jvp-graph (%cj-iree-primal-graph))))
    (%cj-iree-call-with-module
     graph
     (lambda (backend module)
       (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                     (lambda (seed) (%cj-iree-matches-eager-p backend module graph seed 6))
                     :regression-id cond-jvp/iree-jvp-matches-eager
                     :regression-file (regression-path "iree-cond-jvp-matches-eager"
                                                       :package "NABLA.IREE.TESTS"))
           "jvp 変換した cond の IREE の実行結果が eager と一致しなかった")))))

(define-iree-test cond-jvp/iree-vjp-graph-matches-eager
    "cond を vjp 変換した graph（線形な cond を transpose した stablehlo.if を含む）は、IREE で
コンパイル・実行した結果が eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::vjp-graph (%cj-iree-primal-graph))))
    (%cj-iree-call-with-module
     graph
     (lambda (backend module)
       (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                     (lambda (seed) (%cj-iree-matches-eager-p backend module graph seed 4))
                     :regression-id cond-jvp/iree-vjp-matches-eager
                     :regression-file (regression-path "iree-cond-vjp-matches-eager"
                                                       :package "NABLA.IREE.TESTS"))
           "vjp 変換した cond の IREE の実行結果が eager と一致しなかった")))))

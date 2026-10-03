;;;; scan の jvp ルールの性質（issue #135）。
;;;;
;;;; 対象は carry = (h, g, c:i32)、consts = (w)、xs = (u, x2) の scan を含む graph
;;;; （%SCAN-JVP-GRAPH）。入力は (w h g c u x2)、出力は (最終 h, 最終 g, y1, y2)。
;;;;   h' = tanh(h*w + u)   g' = 0.9*g + h   c' = c + 1   y1 = h*u   y2 = 2*x2
;;;; g は h の値を受けるので、g の初期の接線がゼロでも h に接線があれば本体を通って
;;;; 非ゼロになる（carry の接線の不動点）。c は :i32 で常に symbolic zero。
;;;; 期待値は jvp を使わない f64 の中心差分（central-difference-jvp）。接線を渡す入力の
;;;; 部分集合（w h g u x2 のどれに接線があるか）と reverse と長さを PBT で動かす。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %scan-jvp-graph (n length reverse)
  (nb::trace-to-graph
   (nb:with-tracing (w h g c u x2)
     (multiple-value-bind (carry ys)
         (nb:scan (nb:with-tracing (carry x)
                    (let ((h (first carry)) (g (second carry)) (c (third carry))
                          (u (first x)) (x2 (second x)))
                      (values (list (tanh (+ (* h w) u)) (+ (* g 0.9) h) (+ c 1))
                              (list (* h u) (* x2 2.0)))))
                  (list h g c) (list u x2) :length length :reverse reverse)
       (values (first carry) (second carry) (first ys) (second ys))))
   (list (nb:make-aval (list n) :f64) (nb:make-aval (list n) :f64) (nb:make-aval (list n) :f64)
         (nb:make-aval '() :i32)
         (nb:make-aval (list length n) :f64) (nb:make-aval (list length n) :f64))))

(defun %scan-jvp-primals (n length seed)
  "(w h g c u x2) の配列。f64 は入力の範囲を [-1, 1) に収める乱数。"
  (flet ((rnd (shape s) (make-random-array (make-array-spec shape :f64) :seed (+ seed s))))
    (list (rnd (list n) 1) (rnd (list n) 2) (rnd (list n) 3)
          (make-array '() :element-type '(signed-byte 32) :initial-element (mod seed 5))
          (rnd (list length n) 4) (rnd (list length n) 5))))

(defparameter *scan-jvp-float-positions* '(0 1 2 4 5)
  "(w h g c u x2) のうち浮動小数点の入力の位置。")

(defun %scan-jvp-nonzero (mask)
  "MASK（1..31）のビットを、(w h g c u x2) の接線の有無のリスト（c は常に NIL）にする。"
  (let ((bit -1))
    (loop for position below 6
          collect (and (member position *scan-jvp-float-positions*)
                       (logbitp (incf bit) mask)))))

(defun %scan-jvp-tangents (graph nonzero seed)
  "NONZERO が真の位置の接線の乱数配列のリストと、中心差分用の全位置の
（偽の位置は 0 の）接線（c を除いた f64 の入力ぶん）を返す。"
  (let* ((avals (mapcar #'nb:var-aval (nb:graph-invars graph)))
         (given (loop for aval in avals for flag in nonzero for i from 0
                      when flag collect (random-tangent aval :seed (+ seed 100 i))))
         (rest given)
         (full (loop for aval in avals for flag in nonzero for i from 0
                     when (member i *scan-jvp-float-positions*)
                       collect (if flag
                                   (pop rest)
                                   (make-array (nb:aval-shape aval) :element-type 'double-float
                                                                    :initial-element 0d0)))))
    (values given full)))

(defun %scan-jvp-fd (graph primals full-tangents)
  "float の入力だけを動かす中心差分（c は固定）。"
  (let ((c (fourth primals)))
    (central-difference-jvp
     (lambda (w h g u x2) (apply #'nb:eval-graph graph (list w h g c u x2)))
     (loop for p in primals for i from 0 when (member i *scan-jvp-float-positions*) collect p)
     full-tangents)))

(defun %scan-jvp-tangent-outputs (graph nonzero primals given)
  (let* ((jvp (nb::jvp-graph graph :nonzero nonzero))
         (n-out (length (nb:graph-outvars graph)))
         (result (multiple-value-list (apply #'nb:eval-graph jvp (append primals given)))))
    (values (subseq result n-out) (subseq result 0 n-out))))

(test jvp-scan/tangent-matches-central-difference-f64
  "scan の jvp の接線は f64 の中心差分と一致し、主値は元の scan と一致する
（reverse、長さ 0〜4、接線を持つ入力の任意の部分集合を含む）。"
  (is (check-it
       (generator (tuple (integer 0 100000) (integer 0 4) (integer 1 3) (integer 0 1) (integer 1 31)))
       (lambda (case)
         (destructuring-bind (seed length n reverse-code mask) case
           (let* ((graph (%scan-jvp-graph n length (= 1 reverse-code)))
                  (nonzero (%scan-jvp-nonzero mask))
                  (primals (%scan-jvp-primals n length seed)))
             (multiple-value-bind (given full) (%scan-jvp-tangents graph nonzero seed)
               (multiple-value-bind (tangents outs) (%scan-jvp-tangent-outputs graph nonzero primals given)
                 (and (%results-close-p tangents (%scan-jvp-fd graph primals full)
                                        :rtol *autodiff-rtol* :atol *autodiff-atol*)
                      (%results-close-p outs (multiple-value-list (apply #'nb:eval-graph graph primals)))))))))
         :regression-id jvp-scan/tangent-matches-central-difference-f64
         :regression-file (regression-path "jvp-scan-central-difference"))))

(test jvp-scan/tangent-is-linear
  "scan の jvp は接線について線形: jvp(3v) = 3 jvp(v)、jvp(v + w) = jvp(v) + jvp(w)
（接線を持つ入力の部分集合は固定）。"
  (is (check-it
       (generator (tuple (integer 0 100000) (integer 1 4) (integer 1 3) (integer 0 1) (integer 1 31)))
       (lambda (case)
         (destructuring-bind (seed length n reverse-code mask) case
           (let* ((graph (%scan-jvp-graph n length (= 1 reverse-code)))
                  (nonzero (%scan-jvp-nonzero mask))
                  (primals (%scan-jvp-primals n length seed)))
             (flet ((tangent-of (tangents)
                      (%scan-jvp-tangent-outputs graph nonzero primals tangents)))
               (let ((v (%scan-jvp-tangents graph nonzero seed))
                     (w (%scan-jvp-tangents graph nonzero (+ seed 50))))
                 (and (%results-close-p (tangent-of (mapcar (lambda (a) (%scale-array a 3)) v))
                                        (mapcar (lambda (a) (%scale-array a 3)) (tangent-of v))
                                        :rtol 1d-9 :atol 1d-9)
                      (%results-close-p (tangent-of (mapcar #'%sum-array v w))
                                        (mapcar #'%sum-array (tangent-of v) (tangent-of w))
                                        :rtol 1d-9 :atol 1d-9)))))))
         :regression-id jvp-scan/tangent-is-linear
         :regression-file (regression-path "jvp-scan-linear"))))

(defun %jvp-scan-eqn (graph)
  (find :scan (nb:graph-eqns graph) :key (lambda (e) (nb:primitive-name (nb::eqn-prim e)))))

(test jvp-scan/initially-zero-carry-tangent-becomes-nonzero
  "u（xs）にだけ接線があり、g と h の初期の接線はゼロでも、h' = tanh(h w + u) と
g' = 0.9 g + h を通って h と g の carry の接線は不動点で非ゼロになる（回帰テスト）。
jvp した scan の並びは JAX と同じ: consts は w だけ（接線なし）、carry は h g c に
接線 h g が続いて 5、xs は u x2 に u の接線が続いて 3、eqn の invars は 1 + 5 + 3。
接線は中心差分と一致する。"
  (let* ((graph (%scan-jvp-graph 2 3 nil))
         (nonzero '(nil nil nil nil t nil))
         (primals (%scan-jvp-primals 2 3 7))
         (jvp (nb::jvp-graph graph :nonzero nonzero))
         (eqn (%jvp-scan-eqn jvp))
         (params (nb::eqn-params eqn)))
    (is (= 1 (getf params :num-consts)))
    (is (= 5 (getf params :num-carry)))
    (is (= 9 (length (nb::eqn-invars eqn))))
    (is (= 8 (length (nb::eqn-outvars eqn))) "最終 carry 5 + ys は y1 y2 と y1 の接線（y2 は x2 の接線が無いのでゼロ）")
    (multiple-value-bind (given full) (%scan-jvp-tangents graph nonzero 7)
      (is (%results-close-p (%scan-jvp-tangent-outputs graph nonzero primals given)
                            (%scan-jvp-fd graph primals full)
                            :rtol *autodiff-rtol* :atol *autodiff-atol*)))))

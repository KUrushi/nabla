;;;; nb:scan の StableHLO（stablehlo.while + dynamic_slice / dynamic_update_slice）を
;;;; IREE の local backend でコンパイル・実行し、eager（eval-graph）と比べる
;;;; medium テスト（issue #132）。graph ごとに1回コンパイルし、PBT の各試行で再利用する。

(in-package #:nabla.iree.tests)

(defun %scan-iree-arrays (graph seed)
  (loop for var in (nb:graph-invars graph) for i from 0
        for aval = (nb:var-aval var)
        collect (make-random-array (make-array-spec (nb:aval-shape aval) (nb:aval-dtype aval))
                                   :seed (+ seed i))))

(defun %scan-iree-matches-eager-p (backend module graph seed)
  (let* ((arrays (%scan-iree-arrays graph seed))
         (avals (mapcar #'nb:var-aval (nb:graph-invars graph)))
         (device-arrays nil)
         (results nil))
    (unwind-protect
         (progn
           (setf device-arrays (mapcar (lambda (a aval) (to-device a backend :dtype (nb:aval-dtype aval)))
                                       arrays avals))
           (setf results (multiple-value-list
                          (apply #'nabla:backend-invoke backend module "main" device-arrays)))
           (let ((expected (multiple-value-list (apply #'nb:eval-graph graph arrays))))
             (and (= (length results) (length expected))
                  (every (lambda (r e var)
                           (let ((host (to-host r)))
                             (and (equal (array-dimensions host) (array-dimensions e))
                                  (if (eq (nb:aval-dtype (nb:var-aval var)) :i1)
                                      (equalp host e)
                                      (allclose host e :dtype (nb:aval-dtype (nb:var-aval var)))))))
                         results expected (nb:graph-outvars graph)))))
      (dolist (r results) (release-device-array r))
      (dolist (da device-arrays) (release-device-array da)))))

(defmacro %with-scan-iree-check ((graph name) message)
  `(let* ((backend (nabla:find-backend :iree))
          (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo ,graph)))))
     (unwind-protect
          (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                        (lambda (seed) (%scan-iree-matches-eager-p backend module ,graph seed))
                        :regression-id ,(intern (string-upcase (format nil "scan/iree-~A" name)))
                        :regression-file (regression-path ,(format nil "iree-scan-~A" name)
                                                          :package "NABLA.IREE.TESTS"))
              ,message)
       (nabla:backend-unload backend module))))

(defun %scan-iree-mixed-graph (length reverse)
  "carry = (h:f32 [3], c:i32)、xs = (u:[3], v:[2])、ys = (h*u, v, c) の scan を含む graph。"
  (nb::trace-to-graph
   (nb:with-tracing (h c u v)
     (multiple-value-bind (carry ys)
         (nb:scan (nb:with-tracing (carry x)
                    (let ((h (first carry)) (c (second carry)) (u (first x)) (v (second x)))
                      (values (list (tanh (+ (* h 0.5) u)) (+ c 1))
                              (list (* h u) v c))))
                  (list h c) (list u v) :length length :reverse reverse)
       (values (first carry) (second carry) (first ys) (second ys) (third ys))))
   (list (nb:make-aval '(3) :f32) (nb:make-aval '() :i32)
         (nb:make-aval (list length 3) :f32) (nb:make-aval (list length 2) :f32))))

(define-iree-test scan/iree-mixed-forward-matches-eager
    "carry / xs / ys の形と dtype（f32 と i32）が混ざった scan が、IREE の実行結果と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (%scan-iree-mixed-graph 4 nil)))
    (%with-scan-iree-check (graph "matches-lisp-loop") "IREE の scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-mixed-reverse-matches-eager
    "reverse の scan（添字 length-1 から 0）も IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (%scan-iree-mixed-graph 3 t)))
    (%with-scan-iree-check (graph "matches-lisp-loop") "IREE の reverse scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-length-one-matches-eager
    "長さ 1 の scan も IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (%scan-iree-mixed-graph 1 nil)))
    (%with-scan-iree-check (graph "matches-lisp-loop") "IREE の長さ 1 の scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-length-zero-matches-eager
    "長さ 0 の scan（carry は素通し、ys は先頭の軸が 0 の空の配列）も IREE が受け付け、eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (%scan-iree-mixed-graph 0 nil)))
    (%with-scan-iree-check (graph "matches-lisp-loop") "IREE の長さ 0 の scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-without-xs-matches-eager
    "xs の無い scan（length だけ指定）も IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (h)
                  (multiple-value-bind (carry ys)
                      (nb:scan (nb:with-tracing (carry x)
                                 x
                                 (values (list (* (first carry) 0.9)) (list (first carry))))
                               (list h) '() :length 5 :reverse t)
                    (values (first carry) (first ys))))
                (list (nb:make-aval '(2 2) :f32)))))
    (%with-scan-iree-check (graph "without-xs") "IREE の xs 無しの scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-closure-consts-match-eager
    "本体が閉包で捕まえた外側の値（consts）を使う scan も IREE と eager で一致する。
scan の結果を使う後続の演算も含める。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (h0 xs w)
                  (multiple-value-bind (carry ys)
                      (nb:scan (nb:with-tracing (carry x)
                                 (let ((h (tanh (+ (* (first carry) w) (first x)))))
                                   (values (list h) (list (* h w)))))
                               (list h0) (list xs) :reverse t)
                    carry
                    (+ (* (first ys) 2.0) 1.0)))
                (list (nb:make-aval '(3) :f32) (nb:make-aval '(4 3) :f32) (nb:make-aval '(3) :f32)))))
    (%with-scan-iree-check (graph "closure") "IREE の consts つき scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-bf16-carry-and-i1-ys-match-eager
    "bf16 の carry と、x の符号を調べる i1 の ys（IREE は bf16 の演算を融合して丸めの回数が eager と
違いうるので、i1 は丸めに左右されない入力の比較にする）（ys バッファの 0 の初期値のリテラルが dtype ごとに違う）も
IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (h u)
                  (multiple-value-bind (carry ys)
                      (nb:scan (nb:with-tracing (carry x)
                                 (let ((h (+ (first carry) (first x))))
                                   (values (list h) (list h (< (first x) 0.0)))))
                               (list h) (list u))
                    (values (first carry) (first ys) (second ys))))
                (list (nb:make-aval '(3) :bf16) (nb:make-aval '(4 3) :bf16)))))
    (%with-scan-iree-check (graph "bf16-i1") "IREE の bf16 / i1 の scan の結果が eager と一致しなかった")))

;;; ---- IREE 3.11 の Stream AffinityAnalysis のクラッシュの回避（optimization_barrier） ----

(define-iree-test scan/iree-repeated-jit-compiles-do-not-crash
    "while のカウンタと ys の0初期値が constant のままだと、IREE 3.11 の Stream の
AffinityAnalysis が非決定的に落ちる（docs/stablehlo-ops.md）。carry が2つ以上（rank 1 以上を
含む）と ys を持つ scan を、ディスクキャッシュ無しで5回別々にコンパイルして実行しても
落ちず、eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((nb:*compile-cache-directory* nil)
        (backend (nabla:find-backend :iree)))
    (dotimes (i 5)
      (let* ((f (nb:with-tracing (h s xs)
                  (multiple-value-bind (carry ys)
                      (nb:scan (nb:with-tracing (carry x)
                                 (let ((h (first carry)) (s (second carry)) (u (first x)))
                                   (values (list (+ (* h 0.5) u) (+ (* s 0.9) 0.9))
                                           (list (* h 0.5)))))
                               (list h s) (list xs) :reverse t)
                    (values (first carry) (second carry) (first ys)))))
             (h (make-random-array (make-array-spec '(3) :f32) :seed i))
             (s (make-random-array (make-array-spec '() :f32) :seed (+ i 10)))
             (xs (make-random-array (make-array-spec '(4 3) :f32) :seed (+ i 20)))
             (jitted (nb:jit f :backend backend))
             (actual (multiple-value-list (funcall jitted h s xs)))
             (expected (multiple-value-list (funcall f h s xs))))
        (is (every (lambda (a e) (allclose a e :dtype :f32)) actual expected))))))

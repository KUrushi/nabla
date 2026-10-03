;;;; scan を通る逆モード微分の性質（issue #139）。
;;;;
;;;; - vjp-graph の内積テスト（随伴性）: <vjp(u), v> = <u, jvp(v)>、期待値は jvp を使わない
;;;;   f64 の中心差分。#135 の %SCAN-JVP-GRAPH（carry 3 つ・consts・xs 2 つ・:i32 の carry）を使う
;;;; - grad が Elman 型の RNN（W, U, b が閉包で捕まえられた consts）で中心差分と一致する
;;;; - ループ不変な残差が ys に積まれない（linearize した graph の形）
;;;; - scan の transpose は、主値と接線が混ざった scan を拒否する

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %vjp-scan-cotangents (graph seed)
  (loop for outvar in (nb:graph-outvars graph) for i from 0
        collect (random-cotangent (nb:var-aval outvar) :seed (+ seed 500 i))))

(test vjp-scan/adjoint-matches-central-difference-f64
  "scan の vjp は、f64 の中心差分の jvp と随伴: <vjp(u), v> = <u, jvp_fd(v)>
（reverse、長さ 1 を含む 0〜4、接線（余接線）を持つ入力の任意の部分集合）。主値は元の scan と一致する。"
  (is (check-it
       (generator (tuple (integer 0 100000) (integer 0 4) (integer 1 3) (integer 0 1) (integer 1 31)))
       (lambda (case)
         (destructuring-bind (seed length n reverse-code mask) case
           (let* ((graph (%scan-jvp-graph n length (= 1 reverse-code)))
                  (nonzero (%scan-jvp-nonzero mask))
                  (primals (%scan-jvp-primals n length seed))
                  (m (length (nb:graph-outvars graph)))
                  (u (%vjp-scan-cotangents graph seed))
                  (result (multiple-value-list
                           (apply #'nb:eval-graph (nb::vjp-graph graph :nonzero nonzero)
                                  (append primals u))))
                  (cotangents (nthcdr m result)))
             (multiple-value-bind (given full) (%scan-jvp-tangents graph nonzero seed)
               (let ((fd (%scan-jvp-fd graph primals full)))
                 (and (= (length cotangents) (length given))
                      (%results-close-p (subseq result 0 m)
                                        (multiple-value-list (apply #'nb:eval-graph graph primals)))
                      (let ((lhs (reduce #'+ (mapcar #'inner-product cotangents given)))
                            (rhs (reduce #'+ (mapcar #'inner-product u fd))))
                        (<= (abs (- lhs rhs)) (* 1d-6 (+ 1d0 (abs lhs) (abs rhs)))))))))))
       :regression-id vjp-scan/adjoint-matches-central-difference-f64
       :regression-file (regression-path "vjp-scan-adjoint"))))

(defun %elman-loss (length reverse)
  "h' = tanh(W h + U x + b)（W U b は閉包で捕まえた consts）、損失 = 最後の h の総和。"
  (nb:with-tracing (w u b h0 xs)
    (multiple-value-bind (carry ys)
        (nb:scan (nb:with-tracing (carry x)
                   (let ((h (first carry)) (x (first x)))
                     (values (list (tanh (+ (+ (nb:dot w h) (nb:dot u x)) b)))
                             (list h))))
                 (list h0) (list xs) :length length :reverse reverse)
      ys
      (nb:reduce-sum (first carry)))))

(test vjp-scan/elman-rnn-grad-matches-central-difference
  "Elman 型の RNN（勾配が要る consts W U b、carry h、xs）の grad が f64 の中心差分と一致する
（reverse、長さ 1 を含む）。"
  (is (check-it
       (generator (tuple (integer 0 100000) (integer 1 4) (integer 1 3) (integer 0 1)))
       (lambda (case)
         (destructuring-bind (seed length n reverse-code) case
           (let* ((f (%elman-loss length (= 1 reverse-code)))
                  (shapes (list (list n n) (list n n) (list n) (list n) (list length n)))
                  (arrays (loop for shape in shapes for i from 0
                                collect (make-random-array (make-array-spec shape :f64) :seed (+ seed i))))
                  (grads (apply (nb:grad f :argnums '(0 1 2 3 4)) arrays))
                  (expected (central-difference-gradient
                             (lambda (&rest arrays) (funcall f (first arrays) (second arrays) (third arrays)
                                                             (fourth arrays) (fifth arrays)))
                             arrays)))
             (%results-close-p grads expected :rtol *autodiff-rtol* :atol *autodiff-atol*))))
       :regression-id vjp-scan/elman-rnn-grad-matches-central-difference
       :regression-file (regression-path "vjp-scan-elman-grad"))))

(defun %invariant-residual-graph (length)
  "h' = tanh(h * exp(w) + x)。exp(w) は w だけで決まるのでループ不変。"
  (nb::trace-to-graph
   (nb:with-tracing (w h xs)
     (multiple-value-bind (carry ys)
         (nb:scan (nb:with-tracing (carry x)
                    (let ((h (first carry)) (x (first x)))
                      (values (list (tanh (+ (* h (exp w)) x))) (list h))))
                  (list h) (list xs) :length length)
       (values (first carry) (first ys))))
   (list (nb:make-aval '(3) :f64) (nb:make-aval '(3) :f64) (nb:make-aval '(5 3) :f64))))

(defun %scan-eqns (graph)
  (remove-if-not (lambda (e) (eq :scan (nb:primitive-name (nb::eqn-prim e)))) (nb:graph-eqns graph)))

(test vjp-scan/loop-invariant-residuals-are-not-stacked
  "linearize した graph の形: 主値側の scan の出力は、最終 carry・ys に、積んだ残差が
（tanh の微分。ステップごとに変わる。w に接線があれば h も）だけ。exp(w) はループ不変なので ys に積まず、scan の外で
計算して線形な scan の consts として渡す（consts は1つ、carry は1つ）。w に接線がある場合も同じ。"
  (dolist (nonzero '((nil t nil) (t t nil) (t t t)))
    (let* ((graph (%invariant-residual-graph 5))
           (lin (nb::linearize-graph graph :nonzero nonzero))
           (primal-scans (%scan-eqns (nb::linearization-primal-graph lin)))
           (linear-scans (%scan-eqns (nb::linearization-linear-graph lin))))
      (is (= 1 (length primal-scans)))
      (is (= 1 (length linear-scans)))
      ;; 最終 carry 1 + ys 1 + 積んだ残差（tanh の微分。w に接線があれば d(h exp(w)) の h も）
      (is (= (if (first nonzero) 4 3) (length (nb::eqn-outvars (first primal-scans)))))
      (is (equal '(5 3) (nb:aval-shape (nb:var-aval (third (nb::eqn-outvars (first primal-scans)))))))
      (let ((params (nb::eqn-params (first linear-scans))))
        (is (= 1 (getf params :num-carry)))
        ;; ループ不変な残差 exp(w)（+ w に接線があればその接線）。tanh の微分は xs
        (is (= (if (first nonzero) 2 1) (getf params :num-consts))))
      ;; exp(w) の計算は scan の外にある
      (is (find :exp (nb:graph-eqns (nb::linearization-primal-graph lin))
                :key (lambda (e) (nb:primitive-name (nb::eqn-prim e))))))))

(test vjp-scan/transposing-a-mixed-scan-is-rejected
  "主値と接線が混ざった scan（jvp ルールの出力そのもの）は線形ではない。carry が線形入力に
依存しないので、transpose は AUTODIFF-ERROR にする（黙って主値の計算を線形側に入れない）。"
  (let* ((graph (%scan-jvp-graph 2 3 nil))
         (jvp (nb::jvp-graph graph))
         (n (length (nb:graph-invars graph))))
    (signals nb::autodiff-error (nb::transpose-graph jvp n))))

(defparameter *host-constant* (make-array 3 :element-type 'double-float :initial-contents '(0.3d0 -0.2d0 0.5d0)))

(test vjp-scan/closed-over-host-array-is-loop-invariant
  "閉包で捕まえた配列（scan の consts ではなく本体の定数）を使う本体も grad できる:
h' = tanh(h exp(c) + x)（c はホストの配列）。中心差分と一致する。"
  (let* ((f (nb:with-tracing (h0 xs)
              (multiple-value-bind (carry ys)
                  (nb:scan (nb:with-tracing (carry x)
                             (values (list (tanh (+ (* (first carry) (exp *host-constant*)) (first x)))) '()))
                           (list h0) (list xs) :length 4)
                ys
                (nb:reduce-sum (first carry)))))
         (arrays (list (make-random-array (make-array-spec '(3) :f64) :seed 1)
                       (make-random-array (make-array-spec '(4 3) :f64) :seed 2))))
    (is (%results-close-p (apply (nb:grad f :argnums '(0 1)) arrays)
                          (central-difference-gradient (lambda (&rest a) (apply f a)) arrays)
                          :rtol *autodiff-rtol* :atol *autodiff-atol*))))

(test vjp-scan/nested-scan-grad-matches-central-difference
  "scan の本体の中の scan（入れ子の partial eval と transpose）の grad が中心差分と一致する。"
  (let* ((f (nb:with-tracing (w h0 xs)
              (multiple-value-bind (carry ys)
                  (nb:scan (nb:with-tracing (carry x)
                             (let ((inner (nb:scan (nb:with-tracing (c r)
                                                     (values (list (tanh (+ (* (first c) w) (first r)))) '()))
                                                   carry (list (first x)))))
                               (values inner '())))
                           (list h0) (list xs) :length 3)
                ys
                (nb:reduce-sum (first carry)))))
         (arrays (list (make-random-array (make-array-spec '(2) :f64) :seed 3)
                       (make-random-array (make-array-spec '(2) :f64) :seed 4)
                       (make-random-array (make-array-spec '(3 2 2) :f64) :seed 5))))
    (is (%results-close-p (apply (nb:grad f :argnums '(0 1 2)) arrays)
                          (central-difference-gradient (lambda (&rest a) (apply f a)) arrays)
                          :rtol *autodiff-rtol* :atol *autodiff-atol*))))

(test vjp-scan/known-xs-residual-is-forwarded-not-restacked
  "h' = tanh(h * x_t) の接線 dh * x_t は x_t（既知の xs の要素）を残差に使う。これは積み直さず、
外側の xs をそのまま線形な scan の xs に渡す: 主値側の scan の出力は 最終 carry と tanh の微分の残差の2つだけ。"
  (let* ((graph (nb::trace-to-graph
                 (nb:with-tracing (h xs)
                   (first (nb:scan (nb:with-tracing (carry x)
                                     (values (list (tanh (* (first carry) (first x)))) '()))
                                   (list h) (list xs) :length 4)))
                 (list (nb:make-aval '(3) :f64) (nb:make-aval '(4 3) :f64))))
         (lin (nb::linearize-graph graph :nonzero '(t nil)))
         (primal-scan (first (%scan-eqns (nb::linearization-primal-graph lin))))
         (linear-scan (first (%scan-eqns (nb::linearization-linear-graph lin)))))
    (is (= 2 (length (nb::eqn-outvars primal-scan))))
    ;; 線形な scan の xs は x_t と tanh の微分の2つ（consts 0、carry 1、xs 2）
    (is (= 3 (length (nb::eqn-invars linear-scan))))
    (is (= 0 (getf (nb::eqn-params linear-scan) :num-consts)))))

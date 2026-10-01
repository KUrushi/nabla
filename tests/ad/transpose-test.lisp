;;;; nb::transpose-graph と nb::vjp-graph の性質（issue #82）。
;;;;
;;;; 守らせる性質は .claude/skills/nabla-testing/references/properties.md の
;;;; 「自動微分」の節（内積テスト <vjp(u), v> = <u, jvp(v)>）。graph は
;;;; テスト専用プリミティブ（%test-neg / add / reshape / convert / reduce / mul）
;;;; のランダムな f64 の graph。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %sum-inner-products (as bs)
  (reduce #'+ (mapcar #'inner-product as bs)))

(defun %recipe-seed (recipe)
  "RECIPE から決まる、試行ごとに違う seed（同じレシピなら同じ値）。"
  (mod (sxhash (format nil "~S" recipe)) 100000))

(defun %outputs-random-cotangents (graph &key (seed 0))
  (loop for outvar in (nb:graph-outvars graph) for i from 0
        collect (random-cotangent (nb:var-aval outvar) :seed (+ seed 500 i))))

(defun %scalar-close-p (a b)
  (<= (abs (- a b)) (* 1d-9 (+ 1d0 (abs a) (abs b)))))

(test vjp/inner-product-identity
  "内積テスト: ランダムな u, v について <vjp(u), v> = <u, jvp(v)>。多出力・重複した出力・
invar そのもの・定数そのものが出力になる graph を含む。"
  (is (check-it (generator (graph-recipe :dtypes '(:f64) :binary-prims '(:%test-add :%test-mul)))
                (lambda (recipe)
                  (let* ((seed (%recipe-seed recipe))
                         (graph (%mixed-output-graph (build-graph recipe)))
                         (m (length (nb:graph-outvars graph)))
                         (primals (%jvp-arrays graph :seed seed))
                         (v (%jvp-arrays graph :tangent t :seed seed))
                         (u (%outputs-random-cotangents graph :seed seed))
                         (jvp-result (%jvp-eval (nb::jvp-graph graph) (append primals v)))
                         (vjp-result (%jvp-eval (nb::vjp-graph graph) (append primals u)))
                         (cotangents (nthcdr m vjp-result)))
                    (and (equalp (subseq jvp-result 0 m) (subseq vjp-result 0 m))
                         (= (length cotangents) (length v))
                         (%scalar-close-p (%sum-inner-products cotangents v)
                                          (%sum-inner-products u (nthcdr m jvp-result))))))
                :regression-id vjp/inner-product-identity
                :regression-file (regression-path "vjp-inner-product"))))

(test vjp/inner-product-identity-linear-only
  "同じ内積テストを、線形な graph（%test-mul を含まない）で。入力が複数回使われる
graph（add x x）も含む。"
  (is (check-it (generator (graph-recipe :dtypes '(:f64) :max-ops 8))
                (lambda (recipe)
                  (let* ((seed (%recipe-seed recipe))
                         (graph (%mixed-output-graph (build-graph recipe)))
                         (m (length (nb:graph-outvars graph)))
                         (primals (%jvp-arrays graph :seed seed))
                         (v (%jvp-arrays graph :tangent t :seed seed))
                         (u (%outputs-random-cotangents graph :seed seed))
                         (jvp-tangents (nthcdr m (%jvp-eval (nb::jvp-graph graph) (append primals v))))
                         (cotangents (nthcdr m (%jvp-eval (nb::vjp-graph graph) (append primals u)))))
                    (%scalar-close-p (%sum-inner-products cotangents v)
                                     (%sum-inner-products u jvp-tangents))))
                :regression-id vjp/inner-product-identity-linear-only
                :regression-file (regression-path "vjp-inner-product-linear"))))

(test vjp/matches-central-difference-gradient
  "vjp の余接線は、central-difference-gradient（jvp にも transpose にも依存しないオラクル）と
許容誤差で一致する。"
  (is (check-it (generator (graph-recipe :dtypes '(:f64) :max-ops 4 :max-dim 3
                                         :binary-prims '(:%test-add :%test-mul)))
                (lambda (recipe)
                  (let* ((seed (%recipe-seed recipe))
                         (graph (%mixed-output-graph (build-graph recipe)))
                         (m (length (nb:graph-outvars graph)))
                         (primals (%jvp-arrays graph :seed seed))
                         (u (%outputs-random-cotangents graph :seed seed))
                         (cotangents (nthcdr m (%jvp-eval (nb::vjp-graph graph) (append primals u)))))
                    (%results-close-p cotangents (central-difference-gradient graph primals :cotangents u)
                                      :rtol *autodiff-rtol* :atol *autodiff-atol*)))
                :regression-id vjp/matches-central-difference-gradient
                :regression-file (regression-path "vjp-central-difference"))))

(test vjp/result-is-well-formed-and-round-trips
  "vjp-graph の結果は check-graph と print → read → print の往復を満たし、入力は
主値 ++ 出力の余接線、出力は主値の出力 ++ 入力の余接線（aval は元の出力・入力と同じ）。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64) :binary-prims '(:%test-add :%test-mul)))
                (lambda (recipe)
                  (let* ((graph (%mixed-output-graph (build-graph recipe)))
                         (before (nb:print-graph graph))
                         (vjp (nb::vjp-graph graph))
                         (in-avals (mapcar #'nb:var-aval (nb:graph-invars graph)))
                         (out-avals (mapcar #'nb:var-aval (nb:graph-outvars graph))))
                    (and (%jvp-round-trips-p vjp)
                         (equalp (mapcar #'nb:var-aval (nb:graph-invars vjp)) (append in-avals out-avals))
                         (equalp (mapcar #'nb:var-aval (nb:graph-outvars vjp)) (append out-avals in-avals))
                         (string= before (nb:print-graph graph)))))
                :regression-id vjp/result-is-well-formed-and-round-trips
                :regression-file (regression-path "vjp-well-formed"))))

(test vjp/nonzero-selects-input-cotangents
  "nonzero を渡すと、余接線が返るのは nonzero が真の入力だけで、その値は
nonzero 無しの vjp の対応する余接線と一致する。"
  (is (check-it (generator (graph-recipe :dtypes '(:f64) :binary-prims '(:%test-add :%test-mul)))
                (lambda (recipe)
                  (let* ((seed (%recipe-seed recipe))
                         (graph (%mixed-output-graph (build-graph recipe)))
                         (m (length (nb:graph-outvars graph)))
                         (nonzero (%linearize-nonzero graph recipe))
                         (primals (%jvp-arrays graph :seed seed))
                         (u (%outputs-random-cotangents graph :seed seed))
                         (full (nthcdr m (%jvp-eval (nb::vjp-graph graph) (append primals u))))
                         (partial (nthcdr m (%jvp-eval (nb::vjp-graph graph :nonzero nonzero)
                                                       (append primals u)))))
                    (and (= (length partial) (count t nonzero))
                         (%results-close-p partial (loop for c in full for f in nonzero when f collect c)))))
                :regression-id vjp/nonzero-selects-input-cotangents
                :regression-file (regression-path "vjp-nonzero"))))

;;; --- 固定の例（pin） ---

(defun %f64-aval (&rest shape) (nb:make-aval shape :f64))

(defun %f64-array (shape &rest elements)
  (make-array shape :element-type 'double-float :initial-contents elements))

(test transpose/unused-input-gets-instantiated-zero
  "出力に効かない入力の余接線は、ゼロの配列（instantiate-zero）になる。出力が定数だけの
graph の vjp は、主値の出力と、入力と同じ形のゼロ。"
  (let* ((x (nb::make-var (%f64-aval 2 3)))
         (c (nb::make-var (%f64-aval 4)))
         (array (%f64-array '(4) 1d0 2d0 3d0 4d0))
         (graph (nb::make-graph (list x) '() (list c) (list (cons c array))))
         (vjp (nb::vjp-graph graph))
         (result (%jvp-eval vjp (list (make-array '(2 3) :element-type 'double-float :initial-element 5d0)
                                      (make-array '(4) :element-type 'double-float :initial-element 7d0)))))
    (is (%jvp-round-trips-p vjp))
    (is (equalp array (first result)))
    (is (equalp (make-array '(2 3) :element-type 'double-float :initial-element 0d0) (second result)))))

(test transpose/known-only-eqns-are-evaluated-forward
  "transpose-graph は、線形入力に依存しない eqn（既知の値だけの eqn）を順方向に評価して
係数に使う: y = t * (-r) の転置は ct * (-r)。"
  (let* ((r (nb::make-var (%f64-aval 3)))
         (tt (nb::make-var (%f64-aval 3)))
         (neg (nb::make-eqn :%test-neg (list r)))
         (mul (nb::make-eqn :%test-mul (list tt (first (nb:eqn-outvars neg)))))
         (graph (nb::make-graph (list r tt) (list neg mul) (nb:eqn-outvars mul) '()))
         (transposed (nb::transpose-graph graph 1)))
    (is (%jvp-round-trips-p transposed))
    (is (equalp (list (%f64-array '(3) -10d0 40d0 -90d0))
                (%jvp-eval transposed (list (%f64-array '(3) 1d0 -2d0 3d0)
                                            (%f64-array '(3) 10d0 20d0 30d0)))))))

(test transpose/both-operands-linear-signals-autodiff-error
  "線形でない使い方（t * t）は、ルールが autodiff-error にする。"
  (let* ((tt (nb::make-var (%f64-aval 3)))
         (mul (nb::make-eqn :%test-mul (list tt tt)))
         (graph (nb::make-graph (list tt) (list mul) (nb:eqn-outvars mul) '())))
    (signals nb:autodiff-error (nb::transpose-graph graph 0))))

(test transpose/primitive-without-rule-signals-no-transpose-rule
  "transpose ルールの無いプリミティブが線形側にあれば no-transpose-rule（名前つき）。
接線が流れない（nonzero が NIL）なら線形側に現れないので、ルールが無くても vjp は作れる。"
  (let ((graph (%single-eqn-graph :%test-no-transpose)))
    (handler-case (progn (nb::vjp-graph graph) (fail "no-transpose-rule が signal されなかった"))
      (nb:no-transpose-rule (c)
        (is (eq :%test-no-transpose (nb:no-transpose-rule-name c)))))
    (let ((vjp (nb::vjp-graph graph :nonzero '(nil))))
      (is (%jvp-round-trips-p vjp))
      (is (= 2 (length (nb:graph-invars vjp))))
      (is (= 1 (length (nb:graph-outvars vjp)))))))

(test transpose/rule-violating-the-convention-signals-autodiff-error
  "transpose ルールが規約に反した結果（長さの違うリスト、線形入力の余接線が NIL、入力と
aval の違う余接線）を返したら、プリミティブ名の入った autodiff-error。"
  (dolist (mode '(:short :missing :wrong-aval))
    (let ((graph (%single-eqn-graph :%test-bad-transpose :mode mode)))
      (handler-case (progn (nb::vjp-graph graph) (fail "autodiff-error が signal されなかった (~S)" mode))
        (nb:no-transpose-rule () (fail "no-transpose-rule ではなく autodiff-error のはず"))
        (nb:autodiff-error (c)
          (is (search "%TEST-BAD-TRANSPOSE" (princ-to-string c))))))))

(test transpose/zero-cotangent-from-a-rule-is-not-passed-to-the-next-rule
  "ルールが SYMBOLIC-ZERO の余接線を返したら、その var の余接線はゼロとして扱い、
手前の eqn のルールには渡さない。x → neg → zero-ct の vjp は、入力の余接線がゼロの配列。"
  (let* ((x (nb::make-var (%f64-aval 3)))
         (neg (nb::make-eqn :%test-neg (list x)))
         (zero (nb::make-eqn :%test-zero-ct (list (first (nb:eqn-outvars neg)))))
         (graph (nb::make-graph (list x) (list neg zero) (nb:eqn-outvars zero) '()))
         (vjp (nb::vjp-graph graph)))
    (is (%jvp-round-trips-p vjp))
    (is (equalp (list (%f64-array '(3) -1d0 -2d0 -3d0) (%f64-array '(3) 0d0 0d0 0d0))
                (%jvp-eval vjp (list (%f64-array '(3) 1d0 2d0 3d0) (%f64-array '(3) 4d0 5d0 6d0)))))))

(test transpose/rule-returning-a-cotangent-for-a-known-input-signals-autodiff-error
  "既知の入力の位置に NIL でない値を返す transpose ルールは autodiff-error。"
  (let* ((r (nb::make-var (%f64-aval 3)))
         (tt (nb::make-var (%f64-aval 3)))
         (eqn (nb::make-eqn :%test-bad-known-ct (list tt r)))
         (graph (nb::make-graph (list r tt) (list eqn) (nb:eqn-outvars eqn) '())))
    (signals nb:autodiff-error (nb::transpose-graph graph 1))))

(defun %mul-neg-add-graph ()
  "(lambda (x y) (values (- (* x y)) (+ (* x x) y)))（f64 の shape (3)）。"
  (let* ((x (nb::make-var (%f64-aval 3))) (y (nb::make-var (%f64-aval 3)))
         (m (nb::make-eqn :%test-mul (list x y)))
         (n (nb::make-eqn :%test-neg (list (first (nb:eqn-outvars m)))))
         (q (nb::make-eqn :%test-mul (list x x)))
         (a (nb::make-eqn :%test-add (list (first (nb:eqn-outvars q)) y))))
    (nb::make-graph (list x y) (list m n q a)
                    (list (first (nb:eqn-outvars n)) (first (nb:eqn-outvars a))) '())))

(test vjp/second-order-inner-product-identity
  "二階微分: F = mul / neg / add だけの graph、G = (vjp-graph F) に対しても内積テスト
<vjp(G)(u), v> = <u, jvp(G)(v)> が成り立つ（vjp-graph の結果をさらに vjp-graph できる）。"
  (let* ((f (%mul-neg-add-graph))
         (g (nb::vjp-graph f))
         (m (length (nb:graph-outvars g))))
    (is (%jvp-round-trips-p g))
    (dotimes (trial 5)
      (let* ((primals (%jvp-arrays g :seed (* 10 trial)))
             (v (%jvp-arrays g :tangent t :seed (* 10 trial)))
             (u (%outputs-random-cotangents g :seed (* 10 trial)))
             (jvp-result (%jvp-eval (nb::jvp-graph g) (append primals v)))
             (vjp-result (%jvp-eval (nb::vjp-graph g) (append primals u))))
        (is (every (lambda (a b) (equalp a b)) (subseq jvp-result 0 m) (subseq vjp-result 0 m)))
        (is (%scalar-close-p (%sum-inner-products (nthcdr m vjp-result) v)
                             (%sum-inner-products u (nthcdr m jvp-result))))))))

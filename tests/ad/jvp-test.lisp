;;;; nb::jvp-graph の性質（issue #77、77c）。
;;;;
;;;; jvp-graph は内部関数なので nb:: で呼ぶ。graph は tests/graph-recipes.lisp の
;;;; レシピ（%test-neg / %test-add / reshape / convert / reduce と定数）と、
;;;; 実プリミティブ add / neg（with-tracing）から作る。これらはどれも入力に
;;;; ついて線形（定数を足すので全体はアフィン）なので、期待値は jvp-graph を
;;;; 使わない独立なオラクルにする:
;;;;   - 接線: 元の graph の定数をすべて 0 にした graph（= 線形部分）に接線を
;;;;     入れて eval-graph した結果
;;;;   - f64 の graph: central-difference-jvp（tests/support/autodiff.lisp）

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %jvp-arrays (graph &key (seed 0) tangent)
  "GRAPH の各 invar に対する決定的な乱数配列のリスト。TANGENT が真なら接線用
（invar の dtype で、別の seed 系列）。"
  (loop for invar in (nb:graph-invars graph)
        for i from 0
        collect (let ((aval (nb:var-aval invar)))
                  (if tangent
                      (random-tangent aval :seed (+ seed 1000 i) :dtype (nb:aval-dtype aval))
                      (make-random-array (make-array-spec (nb:aval-shape aval) (nb:aval-dtype aval))
                                         :seed (+ seed i))))))

(defun %jvp-eval (graph arrays)
  (multiple-value-list (apply #'nb:eval-graph graph arrays)))

(defun %linear-part (graph)
  "GRAPH の定数をすべてゼロの配列に置き換えた graph（アフィンな graph の線形部分）。"
  (nb::make-graph (nb:graph-invars graph) (nb:graph-eqns graph) (nb:graph-outvars graph)
                  (mapcar (lambda (entry)
                            (let ((array (cdr entry)))
                              (cons (car entry)
                                    (make-array (array-dimensions array)
                                                :element-type (array-element-type array)
                                                :initial-element (coerce 0 (array-element-type array))))))
                          (nb:graph-constants graph))))

(defun %jvp-round-trips-p (graph)
  (and (nb::check-graph graph)
       (let ((text (nb:print-graph graph)))
         (string= text (nb:print-graph (nb::read-graph text))))))

(defun %array-dtype (array)
  (if (eq (array-element-type array) 'double-float) :f64 :f32))

(defun %results-close-p (actual expected &key rtol atol)
  "配列のリスト ACTUAL と EXPECTED が一致する（許容誤差は各配列の dtype の既定、
RTOL / ATOL で上書き）。"
  (and (= (length actual) (length expected))
       (every (lambda (a e) (allclose a e :dtype (%array-dtype a) :rtol rtol :atol atol))
              actual expected)))

(defun %scale-array (array factor)
  (let ((result (make-array (array-dimensions array) :element-type (array-element-type array))))
    (dotimes (i (array-total-size array) result)
      (setf (row-major-aref result i)
            (coerce (* factor (row-major-aref array i)) (array-element-type array))))))

(defun %sum-array (a b)
  (let ((result (make-array (array-dimensions a) :element-type (array-element-type a))))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i) (+ (row-major-aref a i) (row-major-aref b i))))))

(defun %three-output-graph (graph)
  "GRAPH の出力を (out, 最初の invar そのもの, out) に作り直す（複数出力・重複した
outvar・invar をそのまま出力する場合を同時に覆う）。"
  (let ((out (first (last (nb:graph-outvars graph)))))
    (nb::make-graph (nb:graph-invars graph) (nb:graph-eqns graph)
                    (list out (first (nb:graph-invars graph)) out)
                    (nb:graph-constants graph))))

(test jvp/primal-and-tangent-match-original-and-linear-part
  "jvp-graph した graph を eval-graph すると、前半は元の graph の値と一致し、
後半は元の graph の線形部分に接線を適用した値と一致する。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64)))
                (lambda (recipe)
                  (let* ((graph (%three-output-graph (build-graph recipe)))
                         (n-out (length (nb:graph-outvars graph)))
                         (primals (%jvp-arrays graph))
                         (tangents (%jvp-arrays graph :tangent t))
                         (result (%jvp-eval (nb::jvp-graph graph) (append primals tangents))))
                    (and (= n-out 3)
                         (= (length result) (* 2 n-out))
                         (equalp (subseq result 0 n-out) (%jvp-eval graph primals))
                         (%results-close-p (subseq result n-out)
                                           (%jvp-eval (%linear-part graph) tangents)))))
                :regression-id jvp/primal-and-tangent-match-original-and-linear-part
                :regression-file (regression-path "jvp-matches-linear-part"))))

(test jvp/tangent-matches-central-difference-f64
  "f64 の graph では、jvp の接線が central-difference-jvp と許容誤差で一致する。
中心差分のテストが、ルールにも線形部分のオラクルにも依存しない唯一のオラクル。"
  (is (check-it (generator (graph-recipe :dtypes '(:f64)))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (n-out (length (nb:graph-outvars graph)))
                         (primals (%jvp-arrays graph))
                         (tangents (%jvp-arrays graph :tangent t))
                         (result (%jvp-eval (nb::jvp-graph graph) (append primals tangents))))
                    (%results-close-p (subseq result n-out)
                                      (central-difference-jvp graph primals tangents)
                                      :rtol *autodiff-rtol* :atol *autodiff-atol*)))
                :regression-id jvp/tangent-matches-central-difference-f64
                :regression-file (regression-path "jvp-central-difference"))))

(test jvp/tangent-is-linear
  "接線について線形: jvp(a·v) = a·jvp(v)、jvp(v + w) = jvp(v) + jvp(w)（主値は固定）。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64)))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (jvp (nb::jvp-graph graph))
                         (n-out (length (nb:graph-outvars graph)))
                         (primals (%jvp-arrays graph))
                         (v (%jvp-arrays graph :tangent t :seed 0))
                         (w (%jvp-arrays graph :tangent t :seed 50)))
                    (flet ((tangent-of (tangents)
                             (subseq (%jvp-eval jvp (append primals tangents)) n-out))
                           (scaled (arrays) (mapcar (lambda (a) (%scale-array a 3)) arrays))
                           (close-p (actual expected)
                             (%results-close-p actual expected :rtol 1d-4 :atol 1d-4)))
                      (and (close-p (tangent-of (scaled v)) (scaled (tangent-of v)))
                           (close-p (tangent-of (mapcar #'%sum-array v w))
                                    (mapcar #'%sum-array (tangent-of v) (tangent-of w)))))))
                :regression-id jvp/tangent-is-linear
                :regression-file (regression-path "jvp-linear"))))

(test jvp/result-is-well-formed-and-round-trips
  "jvp-graph の結果（定数を含む）は check-graph と print → read → print の往復を
満たし、入出力の aval が規約どおり（主値 ++ 接線）になる。元の graph は変わらない。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64)))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (before (nb:print-graph graph))
                         (jvp (nb::jvp-graph graph))
                         (in-avals (mapcar #'nb:var-aval (nb:graph-invars graph)))
                         (out-avals (mapcar #'nb:var-aval (nb:graph-outvars graph))))
                    (and (%jvp-round-trips-p jvp)
                         (equalp (mapcar #'nb:var-aval (nb:graph-invars jvp)) (append in-avals in-avals))
                         (equalp (mapcar #'nb:var-aval (nb:graph-outvars jvp)) (append out-avals out-avals))
                         ;; 元の定数は残る（ゼロの接線を作る定数が足されうるので >=）。
                         (>= (length (nb:graph-constants jvp)) (length (nb:graph-constants graph)))
                         (string= before (nb:print-graph graph)))))
                :regression-id jvp/result-is-well-formed-and-round-trips
                :regression-file (regression-path "jvp-well-formed"))))

(test jvp/nonzero-drops-inputs-and-treats-them-as-zero
  "nonzero で一部の入力の接線をゼロにすると、その入力の接線は graph の入力から
消え、結果は「その入力の接線に 0 を入れた完全な jvp」と一致する。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64)))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (n (length (nb:graph-invars graph)))
                         (nonzero (loop for i below n collect (evenp (+ i (length recipe)))))
                         (partial (nb::jvp-graph graph :nonzero nonzero))
                         (full (nb::jvp-graph graph))
                         (n-out (length (nb:graph-outvars graph)))
                         (primals (%jvp-arrays graph))
                         (tangents (%jvp-arrays graph :tangent t))
                         (kept (loop for tg in tangents for flag in nonzero when flag collect tg))
                         (zeroed (loop for tg in tangents for flag in nonzero
                                       collect (if flag tg (%scale-array tg 0))))
                         (partial-result (%jvp-eval partial (append primals kept)))
                         (full-result (%jvp-eval full (append primals zeroed))))
                    (and (%jvp-round-trips-p partial)
                         (= (length (nb:graph-invars partial)) (+ n (count t nonzero)))
                         (= (length partial-result) (* 2 n-out))
                         (%results-close-p partial-result full-result))))
                :regression-id jvp/nonzero-drops-inputs-and-treats-them-as-zero
                :regression-file (regression-path "jvp-nonzero"))))

(defun %single-eqn-graph (prim-name &rest params)
  "f64 の shape (2) の1入力に PRIM-NAME を1つ適用する graph。"
  (let* ((x (nb::make-var (nb:make-aval '(2) :f64)))
         (eqn (apply #'nb::make-eqn prim-name (list x) params)))
    (nb::make-graph (list x) (list eqn) (nb:eqn-outvars eqn) '())))

(defun %jvp-eval-no-eager-zero (graph)
  "%test-no-eager は eager 評価できないので、出力のうち接線（instantiate-zero の
定数と broadcast だけ）の部分だけを取り出した graph を評価して返す
（先頭に主値のダミーを置いて、(rest result) が接線になる形）。"
  (let ((tangent-graph (nb::dce-graph
                        (nb::make-graph (nb:graph-invars graph) (nb:graph-eqns graph)
                                        (rest (nb:graph-outvars graph))
                                        (nb:graph-constants graph)))))
    (cons nil (%jvp-eval tangent-graph (list (make-array '(2) :element-type 'double-float))))))

(test jvp/primitive-without-rule-signals-no-jvp-rule
  "jvp ルールの無いプリミティブは no-jvp-rule（名前つき）。ただし接線がすべて
ゼロ（nonzero がすべて NIL）なら、ルールを呼ばないのでエラーにならない。"
  (let ((graph (%single-eqn-graph :%test-no-eager)))
    (handler-case (progn (nb::jvp-graph graph) (fail "no-jvp-rule が signal されなかった"))
      (nb:no-jvp-rule (c)
        (is (eq :%test-no-eager (nb:no-jvp-rule-name c)))))
    (let ((zero (nb::jvp-graph graph :nonzero '(nil))))
      (is (= 1 (length (nb:graph-invars zero))))
      (is (= 2 (length (nb:graph-outvars zero))))
      (is (%jvp-round-trips-p zero))
      (let ((result (%jvp-eval-no-eager-zero zero)))
        (is (every (lambda (a) (every #'zerop (make-array (array-total-size a)
                                                          :displaced-to a :element-type (array-element-type a))))
                   (rest result)))))))

(test jvp/rule-returning-wrong-aval-signals-autodiff-error
  "ルールが主値の出力と aval の違う接線を返したら、プリミティブ名の入った
autodiff-error（no-jvp-rule ではない）。"
  (handler-case (progn (nb::jvp-graph (%single-eqn-graph :%test-bad-jvp))
                       (fail "autodiff-error が signal されなかった"))
    (nb:no-jvp-rule () (fail "no-jvp-rule ではなく autodiff-error のはず"))
    (nb:autodiff-error (c)
      (is (search "%TEST-BAD-JVP" (princ-to-string c))))))

(test jvp/nonzero-length-mismatch-signals-autodiff-error
  "nonzero の長さが graph の入力の個数と違えば autodiff-error。"
  (signals nb:autodiff-error (nb::jvp-graph (%single-eqn-graph :%test-neg) :nonzero '(t t))))

;;; --- 実プリミティブ add / neg ---

(defun %real-add-neg-graph (shape)
  "(lambda (x y) (- (+ x y))) を SHAPE の f64 引数でトレースした graph。"
  (nb:trace-to-graph (nb:with-tracing (x y) (- (+ x y)))
                     (list (nb:make-aval shape :f64) (nb:make-aval shape :f64))))

(defun %shape-of-seed (seed)
  (loop for k from 1 to (mod seed 3) collect (1+ (mod (+ seed k) 3))))

(test jvp/real-add-neg-values-and-tangents
  "実プリミティブ add / neg の graph (- (+ x y)): 値は元の graph と一致し、
接線は中心差分と一致する。nonzero (t nil) なら y の接線を 0 にしたものと一致する。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (let* ((shape (%shape-of-seed seed))
                         (graph (%real-add-neg-graph shape))
                         (jvp (nb::jvp-graph graph))
                         (spec (make-array-spec shape :f64))
                         (x (make-random-array spec :seed seed))
                         (y (make-random-array spec :seed (+ seed 1)))
                         (vx (make-random-array spec :seed (+ seed 2)))
                         (vy (make-random-array spec :seed (+ seed 3)))
                         (result (%jvp-eval jvp (list x y vx vy)))
                         (only-x (%jvp-eval (nb::jvp-graph graph :nonzero '(t nil)) (list x y vx))))
                    (and (%jvp-round-trips-p jvp)
                         (equalp (first result) (first (%jvp-eval graph (list x y))))
                         (%results-close-p (rest result) (central-difference-jvp graph (list x y) (list vx vy))
                                           :rtol *autodiff-rtol* :atol *autodiff-atol*)
                         (%results-close-p (rest only-x)
                                           (central-difference-jvp graph (list x y) (list vx (%scale-array vy 0)))
                                           :rtol *autodiff-rtol* :atol *autodiff-atol*))))
                :regression-id jvp/real-add-neg-values-and-tangents
                :regression-file (regression-path "jvp-real-add-neg"))))

;;; --- inline-graph との組み合わせ（grad の土台） ---

(test jvp/inlined-into-outer-trace-evaluates-like-jvp-graph
  "jvp-graph の結果を、別のトレースの中で inline-graph して（外側の graph にして）
評価しても、jvp-graph の graph を直接評価した結果と一致する。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64)))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (jvp (nb::jvp-graph graph))
                         (outer (nb::%call-with-fresh-trace
                                 (mapcar #'nb:var-aval (nb:graph-invars jvp))
                                 (lambda (&rest tracers)
                                   (values-list (nb::inline-graph jvp tracers)))))
                         (arrays (append (%jvp-arrays graph) (%jvp-arrays graph :tangent t))))
                    (and (%jvp-round-trips-p outer)
                         (equalp (%jvp-eval jvp arrays) (%jvp-eval outer arrays)))))
                :regression-id jvp/inlined-into-outer-trace-evaluates-like-jvp-graph
                :regression-file (regression-path "jvp-inlined"))))

(test jvp/real-add-neg-inlined-via-with-tracing
  "with-tracing / trace-to-graph の中で jvp-graph を inline-graph し、その外側の
graph に外側の演算（neg）を重ねて評価すると、jvp の結果の neg と一致する。"
  (let* ((aval (nb:make-aval '(3) :f64))
         (jvp (nb::jvp-graph (nb:trace-to-graph (nb:with-tracing (x y) (- (+ x y))) (list aval aval))))
         (outer (nb:trace-to-graph
                 (nb:with-tracing (x y vx vy)
                   (let ((results (nb::inline-graph jvp (list x y vx vy))))
                     (values (- (first results)) (- (second results)))))
                 (list aval aval aval aval)))
         (spec (make-array-spec '(3) :f64))
         (arrays (loop for i below 4 collect (make-random-array spec :seed i))))
    (is (%jvp-round-trips-p outer))
    (is (%results-close-p (%jvp-eval outer arrays)
                          (mapcar (lambda (a) (%scale-array a -1)) (%jvp-eval jvp arrays))))))

(test jvp/multi-output-order-is-primals-then-tangents
  "固定例: (values (- x) (+ x y)) の出力は (-x, x+y, -vx, vx+vy) の順（vx ≠ vy）。"
  (let* ((aval (nb:make-aval '(3) :f64))
         (graph (nb:trace-to-graph (nb:with-tracing (x y) (values (- x) (+ x y))) (list aval aval)))
         (spec (make-array-spec '(3) :f64))
         (x (make-random-array spec :seed 1)) (y (make-random-array spec :seed 2))
         (vx (make-random-array spec :seed 3)) (vy (make-random-array spec :seed 4))
         (result (%jvp-eval (nb::jvp-graph graph) (list x y vx vy))))
    (is (= 4 (length result)))
    (is (not (equalp vx vy)))
    (is (%results-close-p result (list (%scale-array x -1) (%sum-array x y)
                                       (%scale-array vx -1) (%sum-array vx vy))))))

;;; --- :i1（接線空間は自明: 接線は常に全 false） ---

(defun %i1-output-graph ()
  ":i1 の定数を出力に持つ graph:
入力 x（f64 (2)）、定数 c（:i1 (2)）、出力 (x, c)。"
  (let* ((x (nb::make-var (nb:make-aval '(2) :f64)))
         (c (nb::make-var (nb:make-aval '(2) :i1)))
         (const (make-array '(2) :element-type 'bit :initial-element 1)))
    (nb::make-graph (list x) '() (list x c) (list (cons c const)))))

(test jvp/i1-output-gets-all-false-i1-tangent
  ":i1 の出力の接線は、同じ shape の全 false の :i1 配列（主値 ++ 接線の個数は保つ）。
emit-stablehlo も通る。"
  (let* ((graph (%i1-output-graph))
         (jvp (nb::jvp-graph graph))
         (result (%jvp-eval jvp (list (make-random-array (make-array-spec '(2) :f64) :seed 1)
                                      (make-random-array (make-array-spec '(2) :f64) :seed 2)))))
    (is (= 4 (length result)))
    (is (%jvp-round-trips-p jvp))
    (is (equalp (nb:aval-dtype (nb:var-aval (fourth (nb:graph-outvars jvp)))) :i1))
    (is (equalp (nb:aval-shape (nb:var-aval (fourth (nb:graph-outvars jvp)))) '(2)))
    (is (equalp (fourth result) (make-array '(2) :element-type 'bit :initial-element 0)))
    (is (search "i1" (nb:emit-stablehlo jvp)))))

(test jvp/nonzero-default-excludes-non-float-inputs
  ":i1 の入力は既定では接線の入力にならず、nonzero に T を渡すと autodiff-error。"
  (let* ((x (nb::make-var (nb:make-aval '(2) :f64)))
         (b (nb::make-var (nb:make-aval '(2) :i1)))
         (graph (nb::make-graph (list x b) '() (list x b) '())))
    (is (= 3 (length (nb:graph-invars (nb::jvp-graph graph)))))
    (signals nb:autodiff-error (nb::jvp-graph graph :nonzero '(t t)))))

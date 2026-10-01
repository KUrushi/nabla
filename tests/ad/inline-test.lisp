;;;; nb::inline-graph / nb::dce-graph の性質（issue #77、77b）。
;;;;
;;;; どちらも内部関数なので nb:: で呼ぶ（tests/eval-test.lisp などと同じ流儀）。
;;;; graph は tests/graph-recipes.lisp のレシピ（test-only の %test-neg /
;;;; %test-add など）から組み立てる。EAGER が f32 / f64 にしか対応して
;;;; いないので、評価を伴う PBT は dtype を f32 / f64 に絞る。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %inline-test-arrays (graph)
  "GRAPH の各 invar に対する決定的な乱数配列のリスト。"
  (loop for invar in (nb:graph-invars graph)
        for i from 0
        collect (make-random-array
                 (make-array-spec (nb:aval-shape (nb:var-aval invar)) (nb:aval-dtype (nb:var-aval invar)))
                 :seed i)))

(defun %inlined (graph)
  "新しいトレースの中で GRAPH をその invar と同じ aval のトレーサへ
INLINE-GRAPH して得た graph を返す。"
  (nb::%call-with-fresh-trace
   (mapcar #'nb:var-aval (nb:graph-invars graph))
   (lambda (&rest tracers)
     (values-list (nb::inline-graph graph tracers)))))

(defun %same-results-p (g1 g2 arrays)
  (equalp (multiple-value-list (apply #'nb:eval-graph g1 arrays))
          (multiple-value-list (apply #'nb:eval-graph g2 arrays))))

(defun %print-read-print-stable-p (graph)
  "GRAPH が check-graph を満たし、print → read → print で文字列が変わらない。"
  (and (nb::check-graph graph)
       (let ((text (nb:print-graph graph)))
         (string= text (nb:print-graph (nb::read-graph text))))))

(defun %neg-array (array)
  "ARRAY（f32 / f64）の要素の符号を反転した新しい配列（%test-neg の独立オラクル）。"
  (let ((result (make-array (array-dimensions array) :element-type (array-element-type array))))
    (dotimes (i (array-total-size array) result)
      (setf (row-major-aref result i) (- (row-major-aref array i))))))

(test inline/evaluates-like-original
  "インライン化した graph を評価した結果は元の graph の結果と一致する。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64)))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (inlined (%inlined graph)))
                    (%same-results-p graph inlined (%inline-test-arrays graph))))
                :regression-id inline/evaluates-like-original
                :regression-file (regression-path "inline-evaluates-like-original"))))

(test inline/result-is-well-formed-and-round-trips
  "インライン化した graph は check-graph と print → read → print の往復を満たし、
eqn の数・出力の数・入力の aval が元と同じになる（再発行なので eqn は増減しない）。"
  (is (check-it (generator (primitive-graph-recipe))
                (lambda (recipe)
                  (let* ((graph (build-primitive-graph recipe))
                         (inlined (%inlined graph)))
                    (and (%print-read-print-stable-p inlined)
                         (= (length (nb:graph-eqns graph)) (length (nb:graph-eqns inlined)))
                         (= (length (nb:graph-outvars graph)) (length (nb:graph-outvars inlined)))
                         (equalp (mapcar #'nb:var-aval (nb:graph-outvars graph))
                                 (mapcar #'nb:var-aval (nb:graph-outvars inlined)))
                         ;; var は新しく作り直される（元の graph と共有しない）。
                         (notany (lambda (v) (member v (nb:graph-invars graph))) (nb:graph-invars inlined)))))
                :regression-id inline/result-is-well-formed-and-round-trips
                :regression-file (regression-path "inline-well-formed"))))

(test inline/into-outer-trace-evaluates-like-original
  "外側のトレースの途中で、外側の演算の結果をインライン化した graph へ渡しても
（入力が外側の eqn の出力のとき）、外側 graph の評価結果は inline-graph を
使わずに graph を eager 評価してから外側の演算を適用した結果と一致する。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64) :max-rank 2))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (avals (mapcar #'nb:var-aval (nb:graph-invars graph)))
                         ;; 外側: 各入力を neg してから graph に渡し、出力を全部 neg する。
                         (outer (nb::%call-with-fresh-trace
                                 avals
                                 (lambda (&rest xs)
                                   (let ((ys (nb::inline-graph
                                              graph
                                              (mapcar (lambda (x) (nb::%trace-eqn :%test-neg (list x))) xs))))
                                     (values-list (mapcar (lambda (y) (nb::%trace-eqn :%test-neg (list y))) ys))))))
                         (arrays (%inline-test-arrays graph))
                         (negated (mapcar (lambda (a) (%neg-array a)) arrays))
                         (expected (mapcar #'%neg-array
                                           (multiple-value-list (apply #'nb:eval-graph graph negated)))))
                    (equalp expected (multiple-value-list (apply #'nb:eval-graph outer arrays)))))
                :regression-id inline/into-outer-trace-evaluates-like-original
                :regression-file (regression-path "inline-into-outer-trace"))))

(test inline/same-graph-twice-gives-independent-vars
  "同じ graph を同じトレースに2回インライン化しても、2回目の結果は1回目と
別の var になり、全体は check-graph を満たす。"
  (let* ((graph (build-graph '((:in :f32 (2)) (:unary :%test-neg 0) (:out 1))))
         (outer (nb::%call-with-fresh-trace
                 (mapcar #'nb:var-aval (nb:graph-invars graph))
                 (lambda (&rest xs)
                   (let ((a (first (nb::inline-graph graph xs)))
                         (b (first (nb::inline-graph graph xs))))
                     (is (not (eq (nb::tracer-var a) (nb::tracer-var b))))
                     (values a b))))))
    (is (= 2 (length (nb:graph-eqns outer))))
    (is (not (eq (first (nb:graph-outvars outer)) (second (nb:graph-outvars outer)))))))

(test inline/constants-are-re-registered
  "graph の定数は現在のトレースの定数として登録され直る（結果 graph に
同じ配列が定数として現れ、評価結果が一致する）。"
  (let* ((graph (build-graph '((:in :f32 (2)) (:const :f32 (2) 7) (:binary :%test-add 0 1) (:out 2))))
         (inlined (%inlined graph)))
    (is (= 1 (length (nb:graph-constants inlined))))
    (is (eq (cdr (first (nb:graph-constants graph))) (cdr (first (nb:graph-constants inlined)))))
    (is (not (eq (car (first (nb:graph-constants graph))) (car (first (nb:graph-constants inlined))))))
    (is (%same-results-p graph inlined (%inline-test-arrays graph)))))

(test inline/arity-mismatch-signals-tracing-error
  "tracers の個数が graph の入力の個数と違えば tracing-error。"
  (let* ((graph (build-graph '((:in :f32 (2)) (:unary :%test-neg 0) (:out 1))))
         (aval (nb:make-aval '(2) :f32)))
    (signals nb:tracing-error
      (nb::%call-with-fresh-trace (list aval aval)
                                  (lambda (x y) (declare (ignore x y))
                                    (values-list (nb::inline-graph graph (list))))))
    (signals nb:tracing-error
      (nb::%call-with-fresh-trace (list aval aval)
                                  (lambda (x y) (values-list (nb::inline-graph graph (list x y))))))))

(test inline/aval-mismatch-signals-tracing-error
  "tracer の aval が対応する入力の aval と違えば（shape が違っても dtype が違っても）
tracing-error。"
  (let ((graph (build-graph '((:in :f32 (2)) (:unary :%test-neg 0) (:out 1)))))
    (dolist (bad (list (nb:make-aval '(3) :f32) (nb:make-aval '(2) :f64) (nb:make-aval '() :f32)))
      (signals nb:tracing-error
        (nb::%call-with-fresh-trace (list bad)
                                    (lambda (x) (values-list (nb::inline-graph graph (list x)))))))))

(test inline/outside-a-trace-signals-tracing-error
  "トレース中でなければ（*current-trace* が NIL）tracing-error。入力も eqn も無い
graph（定数だけ）でも、黙って成功せずに signal する。"
  (signals nb:tracing-error
    (nb::inline-graph (build-graph '((:const :f32 (2) 3) (:out 0))) '())))

(test inline/equal-but-not-eq-aval-is-accepted
  "トレーサの aval が入力の aval と EQUALP なら（同じオブジェクトでなくても）受け付ける。"
  (let ((graph (build-graph '((:in :f32 (2)) (:unary :%test-neg 0) (:out 1)))))
    (is (= 1 (length (nb:graph-eqns
                      (nb::%call-with-fresh-trace
                       (list (nb:make-aval '(2) :f32))
                       (lambda (x) (values-list (nb::inline-graph graph (list x)))))))))))

(test inline/foreign-tracer-signals-tracing-error
  "別のトレースに属するトレーサを渡すと tracing-error（eqn の無い graph でも）。"
  (let ((graph (build-graph '((:in :f32 (2)) (:out 0))))
        (aval (nb:make-aval '(2) :f32))
        (foreign nil))
    (nb::%call-with-fresh-trace (list aval) (lambda (x) (setf foreign x) x))
    (signals nb:tracing-error
      (nb::%call-with-fresh-trace (list aval)
                                  (lambda (x)
                                    ;; 結果は捨てて有効な x を返す: 入口の検査だけが signal できる。
                                    (nb::inline-graph graph (list foreign))
                                    x)))))

;;; --- dce-graph ---

(test dce/preserves-outputs
  "dce-graph した graph は元の graph と同じ結果を返し、invars は EQ のまま残る。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64)))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (dce (nb::dce-graph graph)))
                    (and (%same-results-p graph dce (%inline-test-arrays graph))
                         (equal (nb:graph-invars graph) (nb:graph-invars dce))
                         (equal (nb:graph-outvars graph) (nb:graph-outvars dce)))))
                :regression-id dce/preserves-outputs
                :regression-file (regression-path "dce-preserves-outputs"))))

(test dce/keeps-only-reachable-eqns-and-constants
  "dce-graph 後に残った eqn は元の eqn の（順序を保った）部分列で、すべての
eqn の出力が出力か後続の残った eqn から使われ、すべての定数が使われている。
もう一度 dce-graph しても変わらない。"
  (is (check-it (generator (primitive-graph-recipe))
                (lambda (recipe)
                  (let* ((graph (build-primitive-graph recipe))
                         (dce (nb::dce-graph graph))
                         (used (make-hash-table :test 'eq)))
                    (dolist (v (nb:graph-outvars dce)) (setf (gethash v used) t))
                    (dolist (e (nb:graph-eqns dce))
                      (dolist (v (nb:eqn-invars e)) (setf (gethash v used) t)))
                    (and (nb::check-graph dce)
                         ;; 部分列（EQ の eqn を順に拾える）
                         (let ((rest (nb:graph-eqns graph)))
                           (every (lambda (e)
                                    (let ((m (member e rest)))
                                      (when m (setf rest (cdr m)) t)))
                                  (nb:graph-eqns dce)))
                         (every (lambda (e) (some (lambda (v) (gethash v used)) (nb:eqn-outvars e)))
                                (nb:graph-eqns dce))
                         (every (lambda (entry) (gethash (car entry) used)) (nb:graph-constants dce))
                         (= (length (nb:graph-eqns dce))
                            (length (nb:graph-eqns (nb::dce-graph dce)))))))
                :regression-id dce/keeps-only-reachable-eqns-and-constants
                :regression-file (regression-path "dce-reachable"))))

(test dce/removes-injected-dead-eqns
  "出力に効かない eqn を足した graph を dce-graph すると、足していない graph を
dce-graph した結果と同じ eqn 数・定数数になる。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64)))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (avals (mapcar #'nb:var-aval (nb:graph-invars graph)))
                         (with-dead (nb::%call-with-fresh-trace
                                     avals
                                     (lambda (&rest xs)
                                       ;; 死んだ eqn を2つ（2つ目は1つ目に依存）足してから graph をインライン化する。
                                       (let ((dead (nb::%trace-eqn :%test-neg (list (first xs)))))
                                         (nb::%trace-eqn :%test-neg (list dead)))
                                       (values-list (nb::inline-graph graph xs)))))
                         (clean (nb::dce-graph graph))
                         (cleaned (nb::dce-graph with-dead)))
                    (and (= (+ 2 (length (nb:graph-eqns graph))) (length (nb:graph-eqns with-dead)))
                         (= (length (nb:graph-eqns clean)) (length (nb:graph-eqns cleaned)))
                         (= (length (nb:graph-constants clean)) (length (nb:graph-constants cleaned)))
                         (%same-results-p graph cleaned (%inline-test-arrays graph)))))
                :regression-id dce/removes-injected-dead-eqns
                :regression-file (regression-path "dce-injected-dead"))))

(test dce/drops-dead-eqn-and-dead-constant-but-keeps-unused-invar
  "固定例: 使われない入力は残り、死んだ eqn と死んだ定数は消える。"
  (let* ((aval (nb:make-aval '(2) :f32))
         (graph (nb::%call-with-fresh-trace
                 (list aval aval)
                 (lambda (x y)
                   (declare (ignore y))
                   (let ((c (nb::%lift-array (make-array 2 :element-type 'single-float :initial-element 1.0) x)))
                     (nb::%trace-eqn :%test-add (list x c)))     ; 死んだ eqn（定数 c も死ぬ）
                   (nb::%trace-eqn :%test-neg (list x)))))
         (dce (nb::dce-graph graph)))
    (is (= 2 (length (nb:graph-eqns graph))))
    (is (= 1 (length (nb:graph-constants graph))))
    (is (= 1 (length (nb:graph-eqns dce))))
    (is (= 0 (length (nb:graph-constants dce))))
    (is (= 2 (length (nb:graph-invars dce))))
    (is (%print-read-print-stable-p dce))))

(test dce/result-round-trips
  "dce-graph の結果は check-graph と print → read → print の往復を満たす。"
  (is (check-it (generator (primitive-graph-recipe))
                (lambda (recipe)
                  (%print-read-print-stable-p (nb::dce-graph (build-primitive-graph recipe))))
                :regression-id dce/result-round-trips
                :regression-file (regression-path "dce-round-trips"))))

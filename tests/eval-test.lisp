;;;; nb:eval-graph の性質（issue #39、e0）。
;;;;
;;;; テストは test-only プリミティブ（tests/test-primitives.lisp）と
;;;; graph-recipes.lisp のレシピを使う。EAGER が f32 / f64 にしか
;;;; 対応していないため、PBT は (GRAPH-RECIPE :DTYPES '(:F32 :F64)) で
;;;; bf16 / f16 のレシピを生成しないようにする。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %eval-test-invar-arrays (graph)
  "GRAPH の各 invar に対して、決定的な乱数配列を1つずつ作って返す
（invar の並びの順に、0 始まりの連番をシードにする）。"
  (loop for invar in (nb:graph-invars graph)
        for i from 0
        collect (make-random-array
                 (make-array-spec (nb:aval-shape (nb:var-aval invar)) (nb:aval-dtype (nb:var-aval invar)))
                 :seed i)))

(defun %eval-graph-oracle (graph arrays)
  "EVAL-GRAPH とは独立に GRAPH-EQNS を手で順に PRIMITIVE-EAGER に適用し、
GRAPH-OUTVARS に対応する配列のリストを返す。EVAL-GRAPH の結果と比べる
オラクル。"
  (let ((env (make-hash-table :test 'eq)))
    (loop for invar in (nb:graph-invars graph)
          for array in arrays
          do (setf (gethash invar env) array))
    (dolist (entry (nb:graph-constants graph))
      (setf (gethash (car entry) env) (cdr entry)))
    (dolist (eqn (nb:graph-eqns graph))
      (let* ((prim (nb:eqn-prim eqn))
             (invars (nb:eqn-invars eqn))
             (in-arrays (mapcar (lambda (v) (gethash v env)) invars))
             (in-avals (mapcar #'nb:var-aval invars))
             (result (apply (nb::primitive-eager prim) in-arrays in-avals (nb:eqn-params eqn))))
        (setf (gethash (first (nb:eqn-outvars eqn)) env) result)))
    (mapcar (lambda (v) (gethash v env)) (nb:graph-outvars graph))))

(test eval/matches-independent-oracle
  "EVAL-GRAPH した結果は、同じ graph を手で eqn 順に PRIMITIVE-EAGER に
適用した独立なオラクルの結果と EQUALP で一致する。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64)))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (arrays (%eval-test-invar-arrays graph))
                         (expected (%eval-graph-oracle graph arrays))
                         (actual (multiple-value-list (apply #'nb:eval-graph graph arrays))))
                    (equalp expected actual)))
                :regression-id eval/matches-independent-oracle
                :regression-file (regression-path "eval-matches-independent-oracle"))))

(test eval/output-aval-matches-outvars
  "EVAL-GRAPH の各出力配列の aval（(ARRAY-AVAL ARRAY (AVAL-DTYPE VAR-AVAL))
で決めたもの）は、対応する GRAPH-OUTVARS の var-aval と EQUALP で一致する。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64)))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (arrays (%eval-test-invar-arrays graph))
                         (results (multiple-value-list (apply #'nb:eval-graph graph arrays))))
                    (every (lambda (result outvar)
                             (equalp (nb:array-aval result (nb:aval-dtype (nb:var-aval outvar)))
                                     (nb:var-aval outvar)))
                           results (nb:graph-outvars graph))))
                :regression-id eval/output-aval-matches-outvars
                :regression-file (regression-path "eval-output-aval-matches-outvars"))))

(test eval/deterministic
  "同じ graph・同じ入力を2回 EVAL-GRAPH すると同じ（EQUALP な）結果になる。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64)))
                (lambda (recipe)
                  (let* ((graph (build-graph recipe))
                         (arrays (%eval-test-invar-arrays graph))
                         (r1 (multiple-value-list (apply #'nb:eval-graph graph arrays)))
                         (r2 (multiple-value-list (apply #'nb:eval-graph graph arrays))))
                    (equalp r1 r2)))
                :regression-id eval/deterministic
                :regression-file (regression-path "eval-deterministic"))))

(test eval/arity-mismatch-signals
  "渡した配列の個数が invars と合わないと（多すぎても少なすぎても）
GRAPH-INPUT-MISMATCH。"
  (let* ((a (nb::make-var (nb:make-aval '(2) :f32)))
         (b (nb::make-var (nb:make-aval '(2) :f32)))
         (eqn (nb::make-eqn :%test-add (list a b)))
         (graph (nb::make-graph (list a b) (list eqn) (nb:eqn-outvars eqn)))
         (array (make-random-array (make-array-spec '(2) :f32))))
    (signals nb:graph-input-mismatch (nb:eval-graph graph array))
    (signals nb:graph-input-mismatch (nb:eval-graph graph array array array))))

(test eval/aval-mismatch-signals
  "invar と個数は合っていても、渡した配列の shape・dtype が invar の aval と
食い違うと GRAPH-INPUT-MISMATCH（(UNSIGNED-BYTE 16) の配列を :f32 の invar
に渡した場合も含む）。"
  (let* ((a (nb::make-var (nb:make-aval '(2) :f32)))
         (graph (nb::make-graph (list a) '() (list a))))
    (signals nb:graph-input-mismatch (nb:eval-graph graph (make-random-array (make-array-spec '(3) :f32))))
    (signals nb:graph-input-mismatch (nb:eval-graph graph (make-random-array (make-array-spec '(2) :f64))))
    (signals nb:graph-input-mismatch (nb:eval-graph graph (make-random-array (make-array-spec '(2) :bf16))))))

(test eval/primitive-not-evaluable-signals
  "EAGER を持たないプリミティブ（%test-no-eager）を使う graph を評価すると
PRIMITIVE-NOT-EVALUABLE。NAME はそのプリミティブ名になる。"
  (let* ((a (nb::make-var (nb:make-aval '(2) :f32)))
         (eqn (nb::make-eqn :%test-no-eager (list a)))
         (graph (nb::make-graph (list a) (list eqn) (nb:eqn-outvars eqn))))
    (handler-case
        (progn (nb:eval-graph graph (make-random-array (make-array-spec '(2) :f32)))
               (fail "primitive-not-evaluable が signal されなかった"))
      (nb:primitive-not-evaluable (c)
        (is (eq :%test-no-eager (nb:primitive-not-evaluable-name c)))))))

(test eval/bad-eager-signals-primitive-error
  "abstract-eval と食い違う shape を返す壊れた EAGER（%test-bad-eager）を
使う graph を評価すると、post-eqn の aval 不変量チェックが PRIMITIVE-ERROR
を signal する。"
  (let* ((a (nb::make-var (nb:make-aval '(2) :f32)))
         (eqn (nb::make-eqn :%test-bad-eager (list a)))
         (graph (nb::make-graph (list a) (list eqn) (nb:eqn-outvars eqn))))
    (signals nb:primitive-error (nb:eval-graph graph (make-random-array (make-array-spec '(2) :f32))))))

(test eval/bad-dtype-eager-signals-primitive-error
  "abstract-eval と食い違う要素型（dtype）を返す壊れた EAGER
（%test-bad-dtype-eager）を使う graph を評価すると、post-eqn の aval
不変量チェックが（(ARRAY-AVAL RESULT DTYPE) 自身が signal する
DTYPE-MISMATCH を PRIMITIVE-ERROR にまとめて）PRIMITIVE-ERROR を signal
する。生の DTYPE-MISMATCH を漏らさないことを確かめる。"
  (let* ((a (nb::make-var (nb:make-aval '(2) :f32)))
         (eqn (nb::make-eqn :%test-bad-dtype-eager (list a)))
         (graph (nb::make-graph (list a) (list eqn) (nb:eqn-outvars eqn))))
    (signals nb:primitive-error (nb:eval-graph graph (make-random-array (make-array-spec '(2) :f32))))))

(test eval/undefined-var-signals-malformed-graph
  "CHECK-GRAPH を経由しない、invars にも constants にも無い var を参照する
eqn を持つ graph を評価すると MALFORMED-GRAPH。"
  (let* ((a (nb::make-var (nb:make-aval '(2) :f32)))
         (stray (nb::make-var (nb:make-aval '(2) :f32)))
         (eqn (nb::make-eqn :%test-neg (list stray)))
         (graph (nb::make-graph (list a) (list eqn) (nb:eqn-outvars eqn))))
    (signals nb::malformed-graph (nb:eval-graph graph (make-random-array (make-array-spec '(2) :f32))))))

(test eval/eqn-with-wrong-outvar-count-signals-error
  "eqn の outvars がちょうど1つでない graph（フェーズ1では起こらないはず
だが、手で組み立てれば作れる）を評価すると ERROR を signal する。
0個の場合だけだと、個数の検査を消しても (FIRST '()) の先で別の ERROR に
なって気づけないので、どちらも正しい aval を持つ2個の場合も確かめる
（issue #70 の :delete-form が見つけた抜け）。"
  (let* ((a (nb::make-var (nb:make-aval '(2) :f32)))
         (prim (nb::find-primitive :%test-neg))
         (array (make-random-array (make-array-spec '(2) :f32))))
    (let* ((eqn (nb::%make-eqn prim '() (list a) '()))
           (graph (nb::make-graph (list a) (list eqn) '())))
      (signals error (nb:eval-graph graph array)))
    (let* ((outs (list (nb::make-var (nb:make-aval '(2) :f32))
                       (nb::make-var (nb:make-aval '(2) :f32))))
           (eqn (nb::%make-eqn prim '() (list a) outs))
           (graph (nb::make-graph (list a) (list eqn) (list (first outs)))))
      (signals error (nb:eval-graph graph array)))))

(test eval/outvar-that-is-an-invar-returns-same-array
  "出力 var がそのまま invar である graph は、渡した配列そのもの（EQ）を
返す。"
  (let* ((a (nb::make-var (nb:make-aval '(2) :f32)))
         (graph (nb::make-graph (list a) '() (list a)))
         (array (make-random-array (make-array-spec '(2) :f32))))
    (is (eq array (nb:eval-graph graph array)))))

(test eval/constants-only-graph-returns-constant
  "invars も eqns も無く、outvars が constants の var だけの graph は、
その定数配列そのもの（EQ）を返す。"
  (let* ((array (make-random-array (make-array-spec '(2) :f32)))
         (v (nb::make-var (nb:array-aval array :f32)))
         (graph (nb::make-graph '() '() (list v) (list (cons v array)))))
    (is (eq array (nb:eval-graph graph)))))

(test eval/zero-outputs-returns-no-values
  "outvars が空の graph を評価すると (VALUES) になる。"
  (let ((a (nb::make-var (nb:make-aval '(2) :f32))))
    (is (equal '() (multiple-value-list
                    (nb:eval-graph (nb::make-graph (list a) '() '())
                                   (make-random-array (make-array-spec '(2) :f32))))))))

(test eval/two-outputs-example
  "2つの出力を持つ graph は、それぞれ独立に評価した結果を GRAPH-OUTVARS の
順に多値で返す（形の違う2つの出力にして、順序を取り違える変異を殺す）。"
  (let* ((a (nb::make-var (nb:make-aval '(2) :f32)))
         (neg-eqn (nb::make-eqn :%test-neg (list a)))
         (neg-out (first (nb:eqn-outvars neg-eqn)))
         (reshape-eqn (nb::make-eqn :%test-reshape (list a) :shape '(1 2)))
         (reshape-out (first (nb:eqn-outvars reshape-eqn)))
         (graph (nb::make-graph (list a) (list neg-eqn reshape-eqn) (list neg-out reshape-out)))
         (array (make-random-array (make-array-spec '(2) :f32))))
    (multiple-value-bind (negated reshaped) (nb:eval-graph graph array)
      (is (equal '(2) (array-dimensions negated)))
      (is (equal '(1 2) (array-dimensions reshaped)))
      (is (equalp (- (aref array 0)) (aref negated 0)))
      (is (equalp (aref array 0) (aref reshaped 0 0))))))

(test eval/bf16-f16-invar-round-trips-through-outvar
  "bf16 / f16 の invar をそのまま outvar にした graph は、渡した
(UNSIGNED-BYTE 16) 配列そのもの（EQ）を返す。invar チェックが ARRAY-AVAL を
invar の dtype 付きで呼んで dtype を判別する経路（%EVAL-GRAPH-INVAR-ACTUAL-AVAL）
の正の例。dtype を渡さずに ARRAY-AVAL を呼ぶ壊れた実装だと、u16 配列が既定の
dtype と誤判定されて GRAPH-INPUT-MISMATCH になり、この検査が失敗する。"
  (dolist (dtype '(:bf16 :f16))
    (let* ((a (nb::make-var (nb:make-aval '(2 3) dtype)))
           (graph (nb::make-graph (list a) '() (list a)))
           (array (make-random-array (make-array-spec '(2 3) dtype))))
      (is (eq array (nb:eval-graph graph array))))))

(test eval/bf16-f16-reshape-eqn-succeeds
  "bf16 / f16 の invar に %TEST-RESHAPE の eqn をかけた graph の評価が成功し、
raw なビット列がそのまま出力に写る（%TEST-RESHAPE の EAGER は要素型に依らず
コピーするだけ）。post-eqn の aval 不変量チェック（(ARRAY-AVAL RESULT
(AVAL-DTYPE OUT-AVAL))）も u16 の出力配列に対して成功することを確かめる。"
  (dolist (dtype '(:bf16 :f16))
    (let* ((a (nb::make-var (nb:make-aval '(2 3) dtype)))
           (eqn (nb::make-eqn :%test-reshape (list a) :shape '(3 2)))
           (graph (nb::make-graph (list a) (list eqn) (nb:eqn-outvars eqn)))
           (array (make-random-array (make-array-spec '(2 3) dtype))))
      (let ((result (nb:eval-graph graph array)))
        (is (equal '(3 2) (array-dimensions result)))
        (is (equalp (row-major-aref array 0) (row-major-aref result 0)))
        (is (equalp (row-major-aref array 5) (row-major-aref result 5)))))))

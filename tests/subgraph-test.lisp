;;;; サブグラフを持つ eqn と複数出力の eqn の土台（issue #127、契約 C1 / C2）。
;;;;
;;;; テスト専用の高階プリミティブ %TEST-CALL-SUBGRAPH（tests/support/
;;;; subgraph-primitive.lisp。複数出力・サブグラフ params）を使い、トレース
;;;; （closure conversion）・eval・印字・StableHLO のリージョン・inline / dce /
;;;; jvp / transpose の素通しを確かめる。IREE での実行は tests/iree/subgraph-test.lisp。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defparameter *subgraph-test-avals*
  (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)))

(defun %sg-random-arrays (seed avals)
  (loop for aval in avals for i from 0
        collect (make-random-array (make-array-spec (nb:aval-shape aval) (nb:aval-dtype aval))
                                   :seed (+ seed i))))

(defun %sg-lines (text)
  (loop for start = 0 then (1+ pos)
        for pos = (position #\Newline text :start start)
        collect (subseq text start pos)
        while pos))

(defun %sg-count-lines-matching (needle text)
  (count-if (lambda (line) (search needle line)) (%sg-lines text)))

(defun %sg-prim-names (graph)
  (mapcar (lambda (e) (nb::primitive-name (nb:eqn-prim e))) (nb:graph-eqns graph)))

(defun %sg-body-graph ()
  "a, b から (a+b, a*b) を返す本体を、単体のトレースで graph にしたもの。"
  (nb::trace-to-graph (nb:with-tracing (a b) (values (+ a b) (* a b)))
                      *subgraph-test-avals*))

(defun %sg-call-graph ()
  "本体 (a+b, a*b) を %TEST-CALL-SUBGRAPH 経由で呼ぶ graph。"
  (nb::trace-to-graph
   (nb:with-tracing (x y)
     (let ((r (test-call-subgraph (nb:with-tracing (a b) (values (+ a b) (* a b))) x y)))
       (values (first r) (second r))))
   *subgraph-test-avals*))

;;; ---- 複数出力の eqn（C1） ----

(test subgraph/multiple-output-eqn-has-one-outvar-per-aval
  "複数出力のプリミティブの eqn は、abstract-eval が返した aval ごとに outvar を持つ。"
  (let* ((graph (%sg-call-graph))
         (eqn (first (nb:graph-eqns graph))))
    (is (nb::primitive-multiple-outputs-p (nb:eqn-prim eqn)))
    (is (= 2 (length (nb:eqn-outvars eqn))))
    (is (equalp *subgraph-test-avals* (mapcar #'nb:var-aval (nb:eqn-outvars eqn))))
    (is (not (nb::primitive-multiple-outputs-p (nb::find-primitive :add))))))

(test subgraph/trace-eqn-rejects-multiple-output-primitive
  "%TRACE-EQN（単一出力）に複数出力のプリミティブを渡すと TRACING-ERROR になる。"
  (signals nb::tracing-error
    (nb::trace-to-graph
     (nb:with-tracing (x y)
       (let ((body (nb::trace-to-graph (nb:with-tracing (a b) (values a b)) *subgraph-test-avals*)))
         (nb::%trace-eqn :%test-call-subgraph (list x y) :body body)))
     *subgraph-test-avals*)))

(test subgraph/eager-and-eval-graph-equal-direct-body
  "高階プリミティブ経由の eval-graph は、本体を直接呼んだ結果と一致する（PBT）。"
  (let ((graph (%sg-call-graph))
        (body (%sg-body-graph)))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let ((arrays (%sg-random-arrays seed *subgraph-test-avals*)))
                      (equalp (multiple-value-list (apply #'nb:eval-graph graph arrays))
                              (multiple-value-list (apply #'nb:eval-graph body arrays)))))
                  :regression-id subgraph/eager-equals-body
                  :regression-file (regression-path "subgraph-eager-equals-body")))))

;;; ---- closure conversion（C2） ----

(test subgraph/closure-capture-lifts-outer-tracers-to-extra-inputs
  "本体が閉包で捕まえた外側のトレーサは、サブグラフの追加の入力に持ち上げられ、
eqn の invars の末尾に足される。同じトレーサは1回だけ持ち上げられる。"
  (let* ((graph (nb::trace-to-graph
                 (nb:with-tracing (x y)
                   ;; y を2回使っても追加の入力は1つ（EQ でメモ化）。
                   (first (test-call-subgraph (nb:with-tracing (a) (+ (* a y) y)) x)))
                 *subgraph-test-avals*))
         (eqn (first (nb:graph-eqns graph)))
         (body (getf (nb:eqn-params eqn) :body)))
    (is (= 2 (length (nb:eqn-invars eqn))))
    (is (eq (second (nb:eqn-invars eqn)) (second (nb:graph-invars graph))))
    (is (= 2 (length (nb:graph-invars body))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (destructuring-bind (x y) (%sg-random-arrays seed *subgraph-test-avals*)
                      (allclose (nb:eval-graph graph x y) (reference-add (reference-mul x y) y) :dtype :f32)))))))

(test subgraph/lifted-inputs-follow-first-use-order
  "追加の入力は、本体が最初に使った順に並ぶ。"
  (let* ((graph (nb::trace-to-graph
                 (nb:with-tracing (x y)
                   (first (test-call-subgraph (nb:with-tracing (a) (+ (* a y) x)) x)))
                 *subgraph-test-avals*))
         (eqn (first (nb:graph-eqns graph))))
    ;; 引数 x の後ろに、最初に使われた y、次に x。
    (is (equal (list (first (nb:graph-invars graph))
                     (second (nb:graph-invars graph))
                     (first (nb:graph-invars graph)))
               (nb:eqn-invars eqn)))))

(test subgraph/nested-closure-capture-lifts-through-each-level
  "入れ子のサブグラフで祖先のトレーサを捕まえると、中間のトレースも持ち上げる。"
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (x y)
                  (first (test-call-subgraph
                          (nb:with-tracing (a)
                            (first (test-call-subgraph (nb:with-tracing (b) (+ b y)) a)))
                          x)))
                *subgraph-test-avals*)))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (destructuring-bind (x y) (%sg-random-arrays seed *subgraph-test-avals*)
                      (allclose (nb:eval-graph graph x y) (reference-add x y) :dtype :f32)))))))

(test subgraph/body-may-return-captured-outer-tracer
  "本体が閉包で捕まえた外側のトレーサをそのまま返してもよい（追加の入力に持ち上がる）。"
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (x y)
                  (first (test-call-subgraph (nb:with-tracing (a) (progn a y)) x)))
                *subgraph-test-avals*)))
    (destructuring-bind (x y) (%sg-random-arrays 1 *subgraph-test-avals*)
      (is (equalp y (nb:eval-graph graph x y))))))

(test subgraph/fresh-trace-still-rejects-outer-tracer
  "%CALL-WITH-FRESH-TRACE（grad が使う）は親を持たないので、外側のトレーサを
閉包で捕まえると従来どおり TRACING-ERROR になる。"
  (signals nb::tracing-error
    (nb::trace-to-graph
     (nb:with-tracing (x y)
       (nb::%call-with-fresh-trace (list (nb:make-aval '(2 3) :f32))
                                   (lambda (a) (+ a y))))
     *subgraph-test-avals*)))

(test subgraph/trace-subgraph-requires-traceable-function
  "%TRACE-SUBGRAPH に TRACEABLE-FUNCTION でないものを渡すと TRACING-ERROR。"
  (signals nb::tracing-error (nb::%trace-subgraph #'identity (list (nb:make-aval '() :f32)))))

(test subgraph/trace-subgraph-outside-trace-has-no-captures
  "トレースの外でも %TRACE-SUBGRAPH は使え、captured は空。"
  (multiple-value-bind (graph captured)
      (nb::%trace-subgraph (nb:with-tracing (a) (- a)) (list (nb:make-aval '(3) :f32)))
    (is (= 1 (length (nb:graph-invars graph))))
    (is (null captured))))

;;; ---- 印字 ----

(test subgraph/print-graph-shows-subgraph-eqns-nested
  "サブグラフを持つ graph の印字に、サブグラフの全 eqn が入れ子の深さつきで現れる。"
  (let* ((text (nb:print-graph (%sg-call-graph)))
         (lines (%sg-lines text))
         (call-line (find-if (lambda (l) (search ":= %test-call-subgraph" l)) lines))
         (add-line (find-if (lambda (l) (search ":= add" l)) lines))
         (mul-line (find-if (lambda (l) (search ":= mul" l)) lines)))
    (is (= 1 (%sg-count-lines-matching ":= add" text)))
    (is (= 1 (%sg-count-lines-matching ":= mul" text)))
    (is (and call-line add-line mul-line))
    (flet ((indent (l) (position #\( l)))
      (is (> (indent add-line) (indent call-line)))
      (is (= (indent add-line) (indent mul-line))))))

(test subgraph/print-graph-nests-by-depth
  "さらに入れ子のサブグラフは、もう1段深く字下げされる。"
  (let* ((graph (nb::trace-to-graph
                 (nb:with-tracing (x y)
                   (first (test-call-subgraph
                           (nb:with-tracing (a)
                             (first (test-call-subgraph (nb:with-tracing (b) (- b)) a)))
                           x)))
                 *subgraph-test-avals*))
         (lines (%sg-lines (nb:print-graph graph)))
         (neg-line (find-if (lambda (l) (search ":= neg" l)) lines))
         (call-lines (remove-if-not (lambda (l) (search ":= %test-call-subgraph" l)) lines)))
    (is (= 2 (length call-lines)))
    (is (> (position #\( neg-line) (position #\( (second call-lines))))
    (is (> (position #\( (second call-lines)) (position #\( (first call-lines))))))

(test subgraph/print-read-round-trip-with-subgraph-and-multiple-outputs
  "複数出力の eqn とサブグラフを持つ graph は、印字 → READ-GRAPH → 印字で文字列が一致する。"
  (let* ((graph (%sg-call-graph))
         (text (nb:print-graph graph))
         (read-back (nb::read-graph text)))
    (is (string= text (nb:print-graph read-back)))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let ((arrays (%sg-random-arrays seed *subgraph-test-avals*)))
                      (equalp (multiple-value-list (apply #'nb:eval-graph graph arrays))
                              (multiple-value-list (apply #'nb:eval-graph read-back arrays)))))))))

;;; ---- dce / inline / jvp / transpose の素通し ----

(test subgraph/dce-keeps-eqn-while-any-output-is-used
  "dce-graph は、複数出力の eqn を、出力の1つでも使われる限り消さない。使われなければ消す。"
  (let ((used (nb::trace-to-graph
               (nb:with-tracing (x y)
                 (second (test-call-subgraph
                          (nb:with-tracing (a b) (values (+ a b) (* a b))) x y)))
               *subgraph-test-avals*))
        (unused (nb::trace-to-graph
                 (nb:with-tracing (x y)
                   (test-call-subgraph
                    (nb:with-tracing (a b) (values (+ a b) (* a b))) x y)
                   (- x))
                 *subgraph-test-avals*)))
    (is (equal '(:%test-call-subgraph) (%sg-prim-names (nb::dce-graph used))))
    (is (equal '(:neg) (%sg-prim-names (nb::dce-graph unused))))))

(test subgraph/check-graph-validates-subgraphs
  "サブグラフが壊れていると（外側の graph が正しくても）CHECK-GRAPH が MALFORMED-GRAPH を signal する。"
  (let* ((aval (nb:make-aval '(2) :f32))
         (stray (nb::make-var aval))
         (broken (nb::make-graph '() (list (nb::make-eqn :neg (list stray))) '()))
         (eqn (nb::make-eqn :%test-call-subgraph '() :body broken)))
    (signals nb::malformed-graph
      (nb::check-graph (nb::make-graph '() (list eqn) '())))))

(test subgraph/inline-graph-preserves-subgraph-eqn
  "inline-graph で再発行した graph は、元の graph と同じ結果を返す。"
  (let* ((graph (%sg-call-graph))
         (inlined (nb::trace-to-graph
                   (nb:with-tracing (x y)
                     (values-list (nb::inline-graph graph (list x y))))
                   *subgraph-test-avals*)))
    (is (equal '(:%test-call-subgraph) (%sg-prim-names inlined)))
    (is (= 2 (length (nb:eqn-outvars (first (nb:graph-eqns inlined))))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let ((arrays (%sg-random-arrays seed *subgraph-test-avals*)))
                      (equalp (multiple-value-list (apply #'nb:eval-graph graph arrays))
                              (multiple-value-list (apply #'nb:eval-graph inlined arrays)))))))))

(test subgraph/jvp-with-all-zero-tangents-passes-through
  "接線がすべてゼロなら jvp-graph はルールを呼ばず、サブグラフを持つ eqn を素通しする。
接線があってルールが無ければ NO-JVP-RULE（明示的なエラー）。"
  (let* ((graph (%sg-call-graph))
         (jvp (nb::jvp-graph graph :nonzero '(nil nil))))
    (is (= 1 (count :%test-call-subgraph (%sg-prim-names jvp))))
    (signals nb::no-jvp-rule (nb::jvp-graph graph))))

(test subgraph/transpose-reemits-known-multiple-output-eqn
  "transpose-graph は、線形な入力に依存しない複数出力の eqn を順方向に再発行する。"
  (let* ((graph (nb::trace-to-graph
                 (nb:with-tracing (r tt)
                   (* (first (test-call-subgraph
                                   (nb:with-tracing (a b) (values (+ a b) (* a b))) r r))
                           tt))
                 *subgraph-test-avals*))
         (transposed (nb::transpose-graph graph 1)))
    (is (= 1 (count :%test-call-subgraph (%sg-prim-names transposed))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (destructuring-bind (r ct) (%sg-random-arrays seed *subgraph-test-avals*)
                      (allclose (nb:eval-graph transposed r ct)
                                (reference-mul ct (reference-add r r))
                                :dtype :f32)))))))

;;; ---- StableHLO のリージョン ----

(test subgraph/emit-stablehlo-emits-region-with-prefixed-names
  "サブグラフはリージョンとして出る。リージョン内の SSA 名は %s<k>_ 接頭辞を持ち、
loc は演算の最後の行にだけ付く。"
  (let* ((text (nb:emit-stablehlo (%sg-call-graph)))
         (lines (%sg-lines text)))
    (is (search "\"stablehlo.case\"" text))
    (is (search "%s1_" text))
    (is (= 1 (%sg-count-lines-matching "stablehlo.return %s1_" text)))
    (is (= 1 (count-if (lambda (l) (search "loc(\"eqn-0\")" l))
                       (remove-if-not (lambda (l) (search "tensor<i32>) ->" l)) lines))))
    (is (null (find-if (lambda (l) (and (search "%s1_" l) (search "loc(" l))) lines)))))

(test subgraph/region-counter-gives-each-region-distinct-names
  "同じ graph の中の2つのリージョンは、別の接頭辞（%s1_ と %s2_）を使う。"
  (let* ((graph (nb::trace-to-graph
                 (nb:with-tracing (x y)
                   (let* ((r1 (test-call-subgraph (nb:with-tracing (a b) (+ a b)) x y))
                          (r2 (test-call-subgraph (nb:with-tracing (a b) (* a b)) x (first r1))))
                     (first r2)))
                 *subgraph-test-avals*))
         (text (nb:emit-stablehlo graph)))
    (is (search "%s1_" text))
    (is (search "%s2_" text))))

(test subgraph/region-lines-with-block-arguments
  "arg-names を渡さないと、先頭にブロック引数 ^bb0 を出し、外側の名前には結びつけない。"
  (let ((text (let ((nb::*stablehlo-region-counter* 0))
                (format nil "~{~A~^~%~}" (nb::%stablehlo-region-lines (%sg-body-graph))))))
    (is (search "^bb0(%s1_0: tensor<2x3xf32>, %s1_1: tensor<2x3xf32>):" text))
    (is (search "stablehlo.return" text))))

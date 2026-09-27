;;;; nb:print-graph / nb::read-graph の性質（issue #29、u1b）。
;;;;
;;;; read-graph は内部シンボル（nb::）。テストは内部を nb:: で使っている。
;;;; check-it の生成器は tests/graph-recipes.lisp（u1a）のレシピを共用する
;;;; （BUILD-GRAPH が組み立てた graph そのものは check-it に渡さない。理由は
;;;; graph-recipes.lisp 冒頭のコメントを参照）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %ir-print-var-numbering (graph)
  "GRAPH の var を出現順の番号に写す EQ ハッシュ表を返す（nb::%assign-var-numbers
とは独立に実装した、graph-equal 用のオラクル）。"
  (let ((numbers (make-hash-table :test 'eq))
        (n 0))
    (flet ((assign (v) (setf (gethash v numbers) n) (incf n)))
      (dolist (v (nb:graph-invars graph)) (assign v))
      (dolist (entry (nb:graph-constants graph)) (assign (car entry)))
      (dolist (eqn (nb:graph-eqns graph))
        (dolist (v (nb:eqn-outvars eqn)) (assign v))))
    numbers))

(defun %ir-print-element-equal (dtype a b)
  (if (member dtype '(:f32 :f64)) (= a b) (eql a b)))

(defun %ir-print-const-array-equal (dtype a b)
  (and (equal (array-dimensions a) (array-dimensions b))
       (loop for i below (array-total-size a)
             always (%ir-print-element-equal dtype (row-major-aref a i) (row-major-aref b i)))))

(defun graph-equal (g1 g2)
  "G1 と G2 が、var を出現順の番号に写した上で構造的に等しいかどうかを返す。
avals・プリミティブ名・params・constants の要素（bf16/f16 はビット列、
f32/f64 は = での比較）を比べる。"
  (let ((n1 (%ir-print-var-numbering g1))
        (n2 (%ir-print-var-numbering g2)))
    (labels ((var= (v1 v2)
               (and (equalp (gethash v1 n1) (gethash v2 n2))
                    (equalp (nb:var-aval v1) (nb:var-aval v2))))
             (vars= (l1 l2)
               (and (= (length l1) (length l2)) (every #'var= l1 l2))))
      (and (vars= (nb:graph-invars g1) (nb:graph-invars g2))
           (= (length (nb:graph-constants g1)) (length (nb:graph-constants g2)))
           (every (lambda (e1 e2)
                    (and (var= (car e1) (car e2))
                         (%ir-print-const-array-equal
                          (nb:aval-dtype (nb:var-aval (car e1))) (cdr e1) (cdr e2))))
                  (nb:graph-constants g1) (nb:graph-constants g2))
           (= (length (nb:graph-eqns g1)) (length (nb:graph-eqns g2)))
           (every (lambda (eq1 eq2)
                    (and (eq (nb:primitive-name (nb:eqn-prim eq1)) (nb:primitive-name (nb:eqn-prim eq2)))
                         (equal (nb:eqn-params eq1) (nb:eqn-params eq2))
                         (vars= (nb:eqn-invars eq1) (nb:eqn-invars eq2))
                         (vars= (nb:eqn-outvars eq1) (nb:eqn-outvars eq2))))
                  (nb:graph-eqns g1) (nb:graph-eqns g2))
           (vars= (nb:graph-outvars g1) (nb:graph-outvars g2))))))

(test ir-print/text-round-trips
  "print-graph の出力を read-graph で読み戻し、再度 print-graph すると、
元のテキストと string= で一致する（issue #29 の完了条件）。"
  (is (check-it (generator (graph-recipe))
                (lambda (recipe)
                  (let* ((graph (nb::check-graph (build-graph recipe)))
                         (text (nb:print-graph graph)))
                    (string= text (nb:print-graph (nb::read-graph text)))))
                :regression-id ir-print/text-round-trips
                :regression-file (regression-path "ir-print-text-round-trips"))))

(test ir-print/round-trip-preserves-structure
  "read-graph (print-graph g) は g と構造的に等しい（graph-equal）。"
  (is (check-it (generator (graph-recipe))
                (lambda (recipe)
                  (let ((graph (nb::check-graph (build-graph recipe))))
                    (graph-equal graph (nb::read-graph (nb:print-graph graph)))))
                :regression-id ir-print/round-trip-preserves-structure
                :regression-file (regression-path "ir-print-round-trip-preserves-structure"))))

(test ir-print/three-stream-forms-agree
  "print-graph の3つの呼び方（NIL → 文字列、T → *standard-output*、
ストリーム）が同じテキストを出す。"
  (let* ((v (nb::make-var (nb:make-aval '(2 3) :f32)))
         (graph (nb::make-graph (list v) '() (list v)))
         (text (nb:print-graph graph))
         (stream-text (with-output-to-string (s) (nb:print-graph graph s)))
         (stdout-text (let ((*standard-output* (make-string-output-stream)))
                        (nb:print-graph graph t)
                        (get-output-stream-string *standard-output*))))
    (is (string= text stream-text))
    (is (string= text stdout-text))))

(test ir-print/text-skeleton-matches-graph-shape
  "印字したテキストを read-from-string すると graph で始まり、
:in/:const/:eqns/:out の4節を持ち、要素数が graph と一致する（フォーマットの
骨格だけを固定し、細部はゴールデン文字列にしない）。"
  (is (check-it (generator (graph-recipe))
                (lambda (recipe)
                  (let* ((graph (nb::check-graph (build-graph recipe)))
                         (form (read-from-string (nb:print-graph graph))))
                    (and (string= "GRAPH" (symbol-name (first form)))
                         (= 4 (length (rest form)))
                         (eq :in (first (second form)))
                         (eq :const (first (third form)))
                         (eq :eqns (first (fourth form)))
                         (eq :out (first (fifth form)))
                         (= (length (rest (second form))) (length (nb:graph-invars graph)))
                         (= (length (rest (third form))) (length (nb:graph-constants graph)))
                         (= (length (rest (fourth form))) (length (nb:graph-eqns graph)))
                         (= (length (rest (fifth form))) (length (nb:graph-outvars graph))))))
                :regression-id ir-print/text-skeleton-matches-graph-shape
                :regression-file (regression-path "ir-print-text-skeleton-matches-graph-shape"))))

(defun %ir-print-sample-graph ()
  "%test-neg を1つ適用しただけの、小さな有効な graph を1つ作る（壊れた入力
のテストのベースにする）。"
  (let* ((v (nb::make-var (nb:make-aval '(2 3) :f32)))
         (eqn (nb::make-eqn :%test-neg (list v))))
    (nb::make-graph (list v) (list eqn) (list (first (nb:eqn-outvars eqn))))))

(defun %ir-print-replace-once (string old new)
  "STRING の中で最初に見つかった OLD を NEW に置き換える。壊れたテキストを
「正しい印字結果を1か所置換して作る」ためのヘルパー。"
  (let ((pos (search old string)))
    (assert pos () "テスト用の文字列 ~S が ~S の中に見つからない" old string)
    (concatenate 'string (subseq string 0 pos) new (subseq string (+ pos (length old))))))

(test ir-print/malformed-text-signals-appropriate-condition
  "壊れ方に応じた決まったコンディションが signal される。テキストの改変は
正しい印字結果を1か所置換して作る。"
  (let ((valid (nb:print-graph (%ir-print-sample-graph))))
    (is-true (search "%test-neg" valid) "テスト前提: サンプルの graph に %test-neg が現れるはず")
    (signals nb::graph-syntax-error (nb::read-graph "(foo)"))
    (signals nb::graph-syntax-error
      (nb::read-graph (%ir-print-replace-once valid " (:const)" "")))
    (signals nb::graph-syntax-error
      (nb::read-graph (%ir-print-replace-once valid " := " " ")))
    (signals nb:unknown-primitive
      (nb::read-graph (%ir-print-replace-once valid "%test-neg" "%test-does-not-exist")))
    (signals nb::malformed-graph
      (nb::read-graph (%ir-print-replace-once valid "(:out %1))" "(:out %9))")))
    (signals nb::graph-syntax-error
      (nb::read-graph (%ir-print-replace-once valid "f32 (2 3) :=" "f64 (2 3) :=")))))

(test ir-print/params-containing-t-round-trip
  "params に CL:T が現れる eqn（%test-flag :keep t）は、印字して読み戻すと
(EQ (GETF ... :KEEP) T) が保たれる。READ-GRAPH は *PACKAGE* を
NABLA.GRAPH-SYNTAX（何も USE しない）に束縛して読むので、正規化せずに
放置すると T が別のシンボルとして読まれてしまう。"
  (let* ((v (nb::make-var (nb:make-aval '(2 3) :f32)))
         (eqn (nb::make-eqn :%test-flag (list v) :keep t))
         (graph (nb::make-graph (list v) (list eqn) (list (first (nb:eqn-outvars eqn)))))
         (round-tripped (nb::read-graph (nb:print-graph graph)))
         (read-back-eqn (first (nb:graph-eqns round-tripped))))
    (is (eq t (getf (nb:eqn-params read-back-eqn) :keep)))))

;; issue #29 follow-up (f1)
(test ir-print/print-graph-rejects-undefined-var-reference
  "graph-eqns の invars が invars / constants / 他の eqn の outvars のどこにも
現れない var（未定義参照）を指しているとき、print-graph は \"%nil\" のような
壊れたテキストを出す代わりに MALFORMED-GRAPH を signal する。"
  (let* ((v (nb::make-var (nb:make-aval '(2 3) :f32)))
         (stray (nb::make-var (nb:make-aval '(2 3) :f32)))
         (eqn (nb::make-eqn :%test-neg (list stray)))
         (graph (nb::make-graph (list v) (list eqn) (list (first (nb:eqn-outvars eqn))))))
    (signals nb::malformed-graph (nb:print-graph graph))))

;; issue #29 follow-up (f1)
(test ir-print/read-graph-rejects-trailing-text
  "print-graph の出力の末尾に余分なテキストが付くと GRAPH-SYNTAX-ERROR。"
  (let ((valid (nb:print-graph (%ir-print-sample-graph))))
    (signals nb::graph-syntax-error (nb::read-graph (concatenate 'string valid " garbage")))))

;; issue #29 follow-up (f1)
(test ir-print/read-graph-rejects-duplicate-var-names
  "同じ var 名が :in / :const / :eqns のいずれかで2回定義されているテキストは
MALFORMED-GRAPH（後の定義が前を黙って上書きし、前の var が到達不能になる
壊れたテキストを黙って受け入れない）。3つの節それぞれで境界を確かめる。"
  (signals nb::malformed-graph
    (nb::read-graph "(graph (:in (%0 f32 (2)) (%0 f32 (2))) (:const) (:eqns) (:out %0))"))
  (signals nb::malformed-graph
    (nb::read-graph "(graph (:in) (:const (%0 f32 (2) 1.0 2.0) (%0 f32 (2) 3.0 4.0)) (:eqns) (:out %0))"))
  (signals nb::malformed-graph
    (nb::read-graph
     "(graph (:in (%0 f32 (2))) (:const) (:eqns (%1 f32 (2) := %test-neg () %0) (%1 f32 (2) := %test-neg () %0)) (:out %1))")))

(test ir-print/print-graph-rejects-non-finite-constants
  "NaN を含む f32 定数を持つ graph は print-graph がエラーになる。"
  (let* ((nan (sb-kernel:make-single-float #x7FC00000))
         (array (make-array '(2) :element-type 'single-float :initial-contents (list nan 1.0)))
         (v (nb::make-var (nb:make-aval '(2) :f32)))
         (graph (nb::make-graph '() '() (list v) (list (cons v array)))))
    (signals error (nb:print-graph graph))))

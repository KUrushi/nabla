;;;; trace-test: TRACE-TO-GRAPH / EVAL-GRAPH の性質（issue #32、t1）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %print-graph-string (fn avals)
  "FN（TRACEABLE-FUNCTION）を AVALS でトレースし、PRINT-GRAPH した文字列を
返す（golden テスト用）。"
  (nb:print-graph (nb:trace-to-graph fn avals)))

;;; --- golden: 単純な算術 ---

(test trace/golden-add
  "(+ x y) は :ADD の EQN を1つ持つ graph になる。"
  (is (string= "(graph
 (:in (%0 f32 (2)) (%1 f32 (2)))
 (:const)
 (:eqns
  (%2 f32 (2) := add () %0 %1))
 (:out %2))"
               (%print-graph-string (nb:with-tracing (x y) (+ x y))
                                    (list (nb:make-aval '(2) :f32) (nb:make-aval '(2) :f32))))))

(test trace/golden-unary-minus-is-neg
  "(- x)（単項）は :NEG に書き換わる。"
  (is (string= "(graph
 (:in (%0 f32 (2)))
 (:const)
 (:eqns
  (%1 f32 (2) := neg () %0))
 (:out %1))"
               (%print-graph-string (nb:with-tracing (x) (- x)) (list (nb:make-aval '(2) :f32))))))

(test trace/golden-unary-divide-is-1-over-x
  "(/ x)（単項）は (%T-DIV 1 X) に書き換わり、1.0 の定数を X と同じ shape に
広げてから DIV する。"
  (is (string= "(graph
 (:in (%0 f32 (2)))
 (:const (%1 f32 () 1.0))
 (:eqns
  (%2 f32 (2) := broadcast-in-dim (:shape (2) :dims ()) %1)
  (%3 f32 (2) := div () %2 %0))
 (:out %3))"
               (%print-graph-string (nb:with-tracing (x) (/ x)) (list (nb:make-aval '(2) :f32))))))

(test trace/golden-add-number-rank1-broadcasts
  "(+ x 1) で X の rank が1以上なら、定数を BROADCAST-IN-DIM してから ADD する。"
  (is (string= "(graph
 (:in (%0 f32 (2)))
 (:const (%1 f32 () 1.0))
 (:eqns
  (%2 f32 (2) := broadcast-in-dim (:shape (2) :dims ()) %1)
  (%3 f32 (2) := add () %0 %2))
 (:out %3))"
               (%print-graph-string (nb:with-tracing (x) (+ x 1)) (list (nb:make-aval '(2) :f32))))))

(test trace/golden-add-number-rank0-no-broadcast
  "(+ x 1) で X が rank 0 なら BROADCAST-IN-DIM の EQN を足さない
（%LIFT-NUMBER の rank 0 の分岐）。"
  (is (string= "(graph
 (:in (%0 f32 ()))
 (:const (%1 f32 () 1.0))
 (:eqns
  (%2 f32 () := add () %0 %1))
 (:out %2))"
               (%print-graph-string (nb:with-tracing (x) (+ x 1)) (list (nb:make-aval '() :f32))))))

(test trace/golden-n-ary-sub-is-left-fold
  "(- a b c) は左結合で ((a - b) - c) に畳み込まれる（右からではないことを
pin する。fold-direction の変異対策）。"
  (is (string= "(graph
 (:in (%0 f32 ()) (%1 f32 ()) (%2 f32 ()))
 (:const)
 (:eqns
  (%3 f32 () := sub () %0 %1)
  (%4 f32 () := sub () %3 %2))
 (:out %4))"
               (%print-graph-string (nb:with-tracing (a b c) (- a b c))
                                    (list (nb:make-aval '() :f32) (nb:make-aval '() :f32) (nb:make-aval '() :f32))))))

(test trace/golden-plus-with-no-args-is-zero-constant
  "(+) は0個の EQN、値0.0の定数を出力にする恒等元のテスト。"
  (is (string= "(graph
 (:in)
 (:const (%0 f32 () 0.0))
 (:eqns)
 (:out %0))"
               (%print-graph-string (nb:with-tracing () (+)) '()))))

(test trace/golden-multiple-values-produce-multiple-outvars
  "(VALUES (+ X 1) (- X 1)) は2つの outvars を持つ graph になる。"
  (is (string= "(graph
 (:in (%0 f32 ()))
 (:const (%1 f32 () 1.0) (%2 f32 () 1.0))
 (:eqns
  (%3 f32 () := add () %0 %1)
  (%4 f32 () := sub () %0 %2))
 (:out %3 %4))"
               (%print-graph-string (nb:with-tracing (x) (values (+ x 1) (- x 1))) (list (nb:make-aval '() :f32))))))

(test trace/constant-array-operand-becomes-graph-constant
  "配列を直接リテラルとして渡すと（(NB::%T-ADD X ARRAY) 経由で）、graph の
定数として現れる。"
  (let* ((array (make-array 2 :element-type 'single-float :initial-contents '(1.0 2.0)))
         (f (nb:with-tracing (x) (nb::%t-add x array)))
         (graph (nb:trace-to-graph f (list (nb:make-aval '(2) :f32)))))
    (is (= 1 (length (nb:graph-constants graph))))
    (is (equalp array (cdr (first (nb:graph-constants graph)))))))

(test trace/double-float-literal-on-f32-tracer-uses-tracer-dtype
  "F32 のトレーサに DOUBLE-FLOAT のリテラルを足しても、リフトされる定数は
トレーサの dtype（F32）を使う（リテラル自身の型ではない）。"
  (is (string= "(graph
 (:in (%0 f32 ()))
 (:const (%1 f32 () 1.0))
 (:eqns
  (%2 f32 () := add () %0 %1))
 (:out %2))"
               (%print-graph-string (nb:with-tracing (x) (+ x 1.0d0)) (list (nb:make-aval '() :f32))))))

;;; --- エラー ---

(defun %stash-into (place value)
  "PLACE（1要素の配列）に VALUE を格納して VALUE を返す。トレース対象の
本体からトレーサを外へ持ち出す（SETQ は使えないので、ふつうの関数呼び出し
として使う）ためのテスト用ヘルパー。"
  (setf (aref place 0) value)
  value)

(test trace/tracer-from-finished-trace-signals-tracing-error
  "別の（既に終わった）TRACE-TO-GRAPH のトレーサを、新しいトレースの本体に
持ち込んで使うと TRACING-ERROR になる。"
  (let ((box (make-array 1 :initial-element nil)))
    (nb:trace-to-graph (nb:with-tracing (x) (%stash-into box x)) (list (nb:make-aval '() :f32)))
    (signals nb:tracing-error
      (nb:trace-to-graph (nb:with-tracing (y) (nb::%t-add (aref box 0) y)) (list (nb:make-aval '() :f32))))))

(test trace/body-returning-non-traceable-value-signals-tracing-error
  "トレース対象の関数がトレーサ・実数・配列のいずれでもない値（ここではリスト）
を返すと TRACING-ERROR になる。"
  (signals nb:tracing-error
    (nb:trace-to-graph (nb:with-tracing (x) (list x)) (list (nb:make-aval '() :f32)))))

(test trace/mismatched-avals-signals-primitive-error
  "(+ x y) で X・Y の shape が食い違えば、:ADD の abstract-eval が
PRIMITIVE-ERROR を signal する。"
  (signals nb:primitive-error
    (nb:trace-to-graph (nb:with-tracing (x y) (+ x y))
                        (list (nb:make-aval '(2) :f32) (nb:make-aval '(3) :f32)))))

(test trace/ub16-array-in-eager-signals-dtype-mismatch
  "生の (UNSIGNED-BYTE 16) 配列（bf16/f16 の格納表現）を eager にそのまま
渡すと、dtype が一意に決まらず DTYPE-MISMATCH になる（bf16/f16 を eager で
直接扱えないという既存の制約。CLAUDE.md）。"
  (let ((ub16 (make-array 2 :element-type '(unsigned-byte 16) :initial-element 0)))
    (signals nb:dtype-mismatch (funcall (nb:with-tracing (x y) (+ x y)) ub16 ub16))))

;;; --- / と log は定義域を守った例ベースのテスト（inf/NaN を避ける） ---

(test trace/divide-on-positive-arrays-matches-eager
  "(/ a b) を正の値の配列でトレース・EVAL-GRAPH した結果は、同じ配列を
直接呼んだ結果と一致する（0除算を避けるための例ベース）。"
  (let* ((spec (make-array-spec '(3) :f32))
         (a (make-random-array spec :seed 1 :domain :positive))
         (b (make-random-array spec :seed 2 :domain :positive))
         (f (nb:with-tracing (x y) (/ x y)))
         (graph (nb:trace-to-graph f (list (nb:array-aval a) (nb:array-aval b)))))
    (is (equalp (funcall f a b) (nb:eval-graph graph a b)))))

(test trace/log-on-positive-arrays-matches-eager
  "(log a) を正の値の配列でトレース・EVAL-GRAPH した結果は、直接呼んだ結果と
一致する（負の値・0を避けるための例ベース。負の値での CL の複素数の挙動は
%T-LOG のドキュメントに書く既知の制約）。"
  (let* ((spec (make-array-spec '(3) :f32))
         (a (make-random-array spec :seed 3 :domain :positive))
         (f (nb:with-tracing (x) (log x)))
         (graph (nb:trace-to-graph f (list (nb:array-aval a)))))
    (is (equalp (funcall f a) (nb:eval-graph graph a)))))

;;; --- PBT: ランダムな算術式木のトレース = 直接呼び出し ---
;;;
;;; 式木は :VAR / :LIT の葉と、:UNARY（NEG EXP TANH 1+ 1-）/ :NARY
;;; （+ - * MAX MIN、arity 2 か 3）のノードからなるデータ（レシピ）として
;;; 生成する（check-it が直接 TRACEABLE-FUNCTION を作れないため。
;;; tests/graph-recipes.lisp と同じ考え方）。

(defparameter *expr-unary-ops* '(neg exp tanh 1+ 1-))
(defparameter *expr-nary-ops* '(+ - * max min))

(defun %expr-random-leaf (n-vars)
  "N-VARS個の変数のどれかを指す (:VAR i)、または [-1, 1) の (:LIT v) を返す。"
  (if (zerop (random 2))
      (list :var (random n-vars))
      (list :lit (- (* 2.0d0 (random 1.0d0)) 1.0d0))))

(defun %expr-random-tree (n-vars depth)
  "N-VARS個の変数を使う、深さ高々 DEPTH の式木のレシピをランダムに作る。"
  (cond
    ((or (zerop depth) (zerop (random 3))) (%expr-random-leaf n-vars))
    ((zerop (random 2))
     (list :unary (nth (random (length *expr-unary-ops*)) *expr-unary-ops*)
           (%expr-random-tree n-vars (1- depth))))
    (t
     (let ((arity (+ 2 (random 2))))
       (list* :nary (nth (random (length *expr-nary-ops*)) *expr-nary-ops*)
              (loop repeat arity collect (%expr-random-tree n-vars (1- depth))))))))

(defun %expr-contains-var-p (recipe)
  "RECIPE のどこかに (:VAR ...) の葉があれば真を返す。"
  (ecase (first recipe)
    (:var t)
    (:lit nil)
    (:unary (%expr-contains-var-p (third recipe)))
    (:nary (some #'%expr-contains-var-p (cddr recipe)))))

(defun %expr-random-tree-with-var (n-vars depth)
  "%EXPR-RANDOM-TREE を、少なくとも1つ (:VAR ...) を含む木が出るまで作り直す。
そうでないと木全体がただの数値（トレーサ・配列を一切経由しない値）になり、
配列に直接適用した結果が配列にならず、TRACE-TO-GRAPH の出力（必ず配列に
正規化される。%OUTVAR-OF）と比較できなくなるため。"
  (loop for tree = (%expr-random-tree n-vars depth)
        when (%expr-contains-var-p tree)
          return tree))

(defclass %expr-tree-generator (check-it:generator)
  ((n-vars :initarg :n-vars :reader %expr-tree-n-vars)
   (max-depth :initarg :max-depth :reader %expr-tree-max-depth))
  (:documentation "EXPR-TREE の named generator の実体（%EXPR-RANDOM-TREE-WITH-VAR を包む）。"))

(defmethod check-it:generate ((generator %expr-tree-generator))
  (%expr-random-tree-with-var (%expr-tree-n-vars generator) (%expr-tree-max-depth generator)))

(defmethod check-it:shrink ((generator %expr-tree-generator) test)
  ;; 式木の縮小はしない（tests/graph-recipes.lisp の GRAPH-RECIPE と同じ
  ;; 考え方: 縮小のロジックを別に持つと、それ自体にバグが混ざりうる）。
  (declare (ignore test))
  (check-it:cached-value generator))

(check-it:def-generator expr-tree (&key (n-vars 2) (max-depth 3))
  (make-instance '%expr-tree-generator :n-vars n-vars :max-depth max-depth))

(defun %expr-unary-cl-op (op)
  (ecase op (neg '-) (exp 'cl:exp) (tanh 'cl:tanh) (1+ 'cl:1+) (1- 'cl:1-)))

(defun %expr-to-form (recipe var-symbols)
  "RECIPE（%EXPR-RANDOM-TREE の形式）を、VAR-SYMBOLS を変数に持つ Lisp の
コードに変換する。"
  (ecase (first recipe)
    (:var (nth (second recipe) var-symbols))
    (:lit (second recipe))
    (:unary (list (%expr-unary-cl-op (second recipe)) (%expr-to-form (third recipe) var-symbols)))
    (:nary (list* (second recipe) (mapcar (lambda (r) (%expr-to-form r var-symbols)) (cddr recipe))))))

(defun %expr-var-symbols (n)
  (loop for i below n collect (intern (format nil "V~D" i) '#:nabla.tests)))

(defun %expr-traceable-function (recipe n-vars)
  "RECIPE から (NB:WITH-TRACING (V0 V1 ...) <式>) を組み立てて EVAL し、
TRACEABLE-FUNCTION を返す。"
  (let* ((vars (%expr-var-symbols n-vars))
         (body (%expr-to-form recipe vars)))
    (eval `(nb:with-tracing ,vars ,body))))

(defun %expr-eval-plain (recipe values)
  "RECIPE を、VALUES（VAR インデックスに対応する SINGLE-FLOAT のリスト）で
ふつうの CL として（トレースを介さず）評価する。"
  (labels ((ev (r)
             (ecase (first r)
               (:var (nth (second r) values))
               (:lit (coerce (second r) 'single-float))
               (:unary (funcall (fdefinition (%expr-unary-cl-op (second r))) (ev (third r))))
               (:nary (reduce (fdefinition (second r)) (mapcar #'ev (cddr r)))))))
    (ev recipe)))

(defun %scalar-bits-equal-p (a b)
  "A・B（同じ浮動小数点型）がビット列として一致するか判定する（NaN=NaN、
符号付き0を区別する）。"
  (etypecase a
    (single-float (= (sb-kernel:single-float-bits a) (sb-kernel:single-float-bits b)))
    (double-float (and (= (sb-kernel:double-float-high-bits a) (sb-kernel:double-float-high-bits b))
                        (= (sb-kernel:double-float-low-bits a) (sb-kernel:double-float-low-bits b))))))

(defun %array-bits-equal-p (a b)
  "A・B（同じ shape・浮動小数点要素型の配列）の全要素がビット列として一致
するか判定する。"
  (and (equal (array-dimensions a) (array-dimensions b))
       (dotimes (i (array-total-size a) t)
         (unless (%scalar-bits-equal-p (row-major-aref a i) (row-major-aref b i))
           (return nil)))))

(test trace/expr-tree-eval-graph-matches-direct-call-on-arrays
  "ランダムな算術式木を TRACE-TO-GRAPH + EVAL-GRAPH した結果は、
TRACEABLE-FUNCTION を配列に直接適用した結果とビット単位で一致する
（同じ eager カーネルを同じ順序で呼ぶので丸め誤差すら生じないはず）。"
  (is (check-it (generator (tuple (expr-tree :n-vars 2 :max-depth 3)
                                   (array-spec :dtypes '(:f32 :f64) :max-rank 3 :max-dim 4)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (recipe spec seed) args
                    (let* ((f (%expr-traceable-function recipe 2))
                           (a (make-random-array spec :seed seed))
                           (b (make-random-array spec :seed (1+ seed)))
                           (direct (funcall f a b))
                           (graph (nb:trace-to-graph f (list (nb:array-aval a) (nb:array-aval b))))
                           (traced (nb:eval-graph graph a b)))
                      (%array-bits-equal-p direct traced))))
                :regression-id trace/expr-tree-eval-graph-matches-direct-call-on-arrays
                :regression-file (regression-path "trace-expr-tree-matches-eval-graph"))))

(test trace/expr-tree-direct-call-matches-plain-cl-eval-on-scalars
  "ランダムな算術式木を SINGLE-FLOAT のスカラーに直接適用した結果は、
%EXPR-EVAL-PLAIN（トレースを介さないふつうの CL の評価）と（許容誤差つきで）
一致する。"
  (is (check-it (generator (tuple (expr-tree :n-vars 2 :max-depth 3)
                                   (uniform-real :lo -1.0d0 :hi 1.0d0)
                                   (uniform-real :lo -1.0d0 :hi 1.0d0)))
                (lambda (args)
                  (destructuring-bind (recipe x y) args
                    (let* ((f (%expr-traceable-function recipe 2))
                           (vx (coerce x 'single-float))
                           (vy (coerce y 'single-float))
                           (direct (funcall f vx vy))
                           (plain (%expr-eval-plain recipe (list vx vy))))
                      (approx= direct plain :dtype :f32))))
                :regression-id trace/expr-tree-direct-call-matches-plain-cl-eval-on-scalars
                :regression-file (regression-path "trace-expr-tree-matches-plain-eval"))))

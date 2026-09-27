;;;; trace-if-test: IF を SELECT に書き換える %T-IF / %T-SELECT の性質
;;;; （issue #32、t2）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; --- golden: if -> compare + neg + select ---

(test trace-if/compare-then-select
  "(IF (< X 0) (- X) X) は :COMPARE + :NEG + :SELECT の3つの EQN になる
（THEN/ELSE を両方トレースしてから SELECT で選ぶ）。"
  (is (string= "(graph
 (:in (%0 f32 ()))
 (:const (%1 f32 () 0.0))
 (:eqns
  (%2 i1 () := compare (:direction :lt) %0 %1)
  (%3 f32 () := neg () %0)
  (%4 f32 () := select () %2 %3 %0))
 (:out %4))"
               (%print-graph-string (nb:with-tracing (x) (if (< x 0) (- x) x))
                                    (list (nb:make-aval '() :f32))))))

(test trace-if/when-with-tracer-test-signals-tracing-error
  "(WHEN P X) は (IF P X NIL) に展開される。ELSE が省略されると NIL（Lisp
の偽）になり、数値としてリフトできないので、P がトレーサ（:I1）の条件
だと TRACING-ERROR になる（契約に明記されたドキュメント上の既知の制約）。"
  (signals nb:tracing-error
    (nb:trace-to-graph (nb:with-tracing (p x) (when p x))
                        (list (nb:make-aval '() :i1) (nb:make-aval '() :f32)))))

(test trace-if/unless-with-tracer-test-signals-tracing-error
  "(UNLESS P X) は (IF P NIL X) に展開される。同じ理由（ELSE 側が NIL）で
TRACING-ERROR になる。"
  (signals nb:tracing-error
    (nb:trace-to-graph (nb:with-tracing (p x) (unless p x))
                        (list (nb:make-aval '() :i1) (nb:make-aval '() :f32)))))

(test trace-if/cond-with-tracer-test-uses-select
  "(COND ((< X 0) (- X)) (T X)) は COND -> IF に展開され、SELECT になる。"
  (let* ((f (nb:with-tracing (x) (cond ((< x 0) (- x)) (t x))))
         (graph (nb:trace-to-graph f (list (nb:make-aval '() :f32)))))
    (is (find :select (nb:graph-eqns graph) :key (lambda (e) (nb:primitive-name (nb:eqn-prim e)))))))

(test trace-if/case-expands-through-cond-and-if
  "CASE は COND を経て IF に展開されるマクロの一例。KEY はトレーサでない
ふつうの値なので分岐そのものは普通の IF のままだが、選ばれた分岐の中の
トレーサ演算（(- X)）は正しくトレースされる。"
  (let* ((f (nb:with-tracing (x) (case (if (> 1 0) 0 1) (0 (- x)) (t x)))))
    (is (= -3.0 (funcall f 3.0)))))

(test trace-if/eager-array-pred-with-tracer-branch-lifts-pred
  "PRED が eager な bit 配列でも、A・B の少なくとも一方がトレーサなら
PRED を :I1 の定数としてリフトしてトレースする（PRED を配列のまま
%EAGER-SELECT-ARRAY に渡すと、トレーサを配列演算に落とそうとして
SIMPLE-TYPE-ERROR で死ぬ回帰）。"
  (let* ((f (nb:with-tracing (x)
              (if (> (make-array 2 :element-type 'single-float :initial-contents '(1.0 -1.0)) 0.0)
                  x
                  0.0)))
         (graph (nb:trace-to-graph f (list (nb:make-aval '(2) :f32))))
         (x (make-array 2 :element-type 'single-float :initial-contents '(3.0 4.0))))
    (is (find :select (nb:graph-eqns graph) :key (lambda (e) (nb:primitive-name (nb:eqn-prim e)))))
    (is (equalp #(3.0 0.0) (funcall f x)))
    (is (equalp #(3.0 0.0) (nb:eval-graph graph x)))))

(test trace-if/ordinary-if-on-lisp-boolean-produces-no-select
  "TEST がトレーサ・配列でない、ふつうの Lisp の値なら SELECT の EQN を
足さない（THEN／ELSE の片方だけを評価する、ふつうの IF のまま）。"
  (let* ((f (nb:with-tracing (x) (if (> 1 0) (+ x 1) (+ x 2))))
         (graph (nb:trace-to-graph f (list (nb:make-aval '() :f32)))))
    (is (null (find :select (nb:graph-eqns graph) :key (lambda (e) (nb:primitive-name (nb:eqn-prim e))))))))

;;; --- and / or on tracers: (if #:g #:g else) の :i1 分岐は tracing-error ---

(test trace-if/and-on-tracers-signals-tracing-error
  "(AND (< X 0) (< X 1)) は (IF #:G #:G (< X 1)) に展開され、:I1 の #:G が
そのまま SELECT の分岐に来るので TRACING-ERROR になる。"
  (signals nb:tracing-error
    (nb:trace-to-graph (nb:with-tracing (x) (and (< x 0) (< x 1)))
                        (list (nb:make-aval '() :f32)))))

(test trace-if/or-on-tracers-signals-tracing-error
  "(OR (< X 0) (< X 1)) も同様に TRACING-ERROR になる。"
  (signals nb:tracing-error
    (nb:trace-to-graph (nb:with-tracing (x) (or (< x 0) (< x 1)))
                        (list (nb:make-aval '() :f32)))))

;;; --- エラー: 両方数値、:i1 でないトレーサの条件 ---

(test trace-if/both-number-branches-signals-tracing-error
  "(IF P 1 2)（P がトレーサ、両方の分岐が数値）は dtype を決められないので
TRACING-ERROR になる。"
  (signals nb:tracing-error
    (nb:trace-to-graph (nb:with-tracing (p) (if p 1.0 2.0)) (list (nb:make-aval '() :i1)))))

(test trace-if/non-i1-tracer-test-signals-tracing-error
  "TEST が :I1 でない dtype のトレーサだと TRACING-ERROR になる。"
  (signals nb:tracing-error
    (nb:trace-to-graph (nb:with-tracing (p x) (if p x 0.0))
                        (list (nb:make-aval '() :f32) (nb:make-aval '() :f32)))))

(test trace-if/number-branch-lifts-to-other-branchs-dtype-and-shape
  "(IF P X 0) は 0 を X と同じ dtype・shape にリフトしてから SELECT する。"
  (is (string= "(graph
 (:in (%0 i1 (2)) (%1 f32 (2)))
 (:const (%2 f32 () 0.0))
 (:eqns
  (%3 f32 (2) := broadcast-in-dim (:shape (2) :dims ()) %2)
  (%4 f32 (2) := select () %0 %1 %3))
 (:out %4))"
               (%print-graph-string (nb:with-tracing (p x) (if p x 0.0))
                                    (list (nb:make-aval '(2) :i1) (nb:make-aval '(2) :f32))))))

;;; --- PBT: 算術式木 + IF/SELECT を含む式木の traced-vs-direct 一致 ---
;;;
;;; t1 の EXPR-TREE 生成器を、:IF ノード（(<CMP> E E) を条件にする）を
;;; 追加した形に拡張する。P と Q に同じ式（E <= E）を渡す組み合わせも
;;; 生成できるようにして、< と <= の境界（タイの扱い）を区別する変異体を
;;; 殺す。

(defparameter *if-compare-directions* '(< <= > >= = /=))

(defun %expr-random-tree-if (n-vars depth)
  "%EXPR-RANDOM-TREE-WITH-VAR を、IF/COMPARE ノードも生成できるように
拡張したバージョン。DEPTH が尽きたら通常の葉、そうでなければ 1/4 の
確率で IF ノードを、それ以外は t1 の %EXPR-RANDOM-TREE と同じ分布で作る。
(:IF <cmp> <test-lhs> <test-rhs> <then> <else>) の形。

THEN／ELSE は（IF をネストさせず）常に %EXPR-RANDOM-TREE-WITH-VAR（t1、
必ず (:VAR ...) を含む）で作る。IF の THEN／ELSE が両方とも数値（配列を
経由しない定数式）になると SELECT/WHERE の dtype が決められず
TRACING-ERROR になる（%T-SELECT の仕様どおり）ため、THEN／ELSE には必ず
配列（トレース時はトレーサ）を経由する式を選び、この失敗を作らないように
する。TEST-LHS／TEST-RHS（比較の被演算子）は数値どうしの比較でも問題ない
ので、そちらだけ IF を再帰的にネストできる。"
  (cond
    ((zerop depth) (%expr-random-leaf n-vars))
    ((zerop (random 4))
     (let ((cmp (nth (random (length *if-compare-directions*)) *if-compare-directions*))
           (test-lhs (%expr-random-tree-if n-vars (1- depth))))
       (list :if cmp test-lhs
             ;; タイを踏むケースも生成するため、右辺は半分の確率で TEST-LHS
             ;; と同じ木を再利用する（(<= E E) のような自己比較）。
             (if (zerop (random 2)) test-lhs (%expr-random-tree-if n-vars (1- depth)))
             (%expr-random-tree-with-var n-vars (1- depth))
             (%expr-random-tree-with-var n-vars (1- depth)))))
    (t (let ((base (%expr-random-tree n-vars depth)))
         (if (eq (first base) :var)
             ;; %EXPR-RANDOM-TREE の葉は :VAR/:LIT のみ返しうる。IF 経路も
             ;; 混ぜたいのでここでは非葉だけ受け入れ直す。
             (%expr-random-tree-if n-vars depth)
             base)))))

(defun %expr-if-contains-var-p (recipe)
  (ecase (first recipe)
    (:var t)
    (:lit nil)
    (:unary (%expr-if-contains-var-p (third recipe)))
    (:nary (some #'%expr-if-contains-var-p (cddr recipe)))
    (:if (destructuring-bind (op test-lhs test-rhs then else) (rest recipe)
           (declare (ignore op))
           (some #'%expr-if-contains-var-p (list test-lhs test-rhs then else))))))

(defun %expr-random-tree-if-with-var (n-vars depth)
  (loop for tree = (%expr-random-tree-if n-vars depth)
        when (%expr-if-contains-var-p tree)
          return tree))

(defclass %expr-if-tree-generator (check-it:generator)
  ((n-vars :initarg :n-vars :reader %expr-if-tree-n-vars)
   (max-depth :initarg :max-depth :reader %expr-if-tree-max-depth)))

(defmethod check-it:generate ((generator %expr-if-tree-generator))
  (%expr-random-tree-if-with-var (%expr-if-tree-n-vars generator) (%expr-if-tree-max-depth generator)))

(defmethod check-it:shrink ((generator %expr-if-tree-generator) test)
  (declare (ignore test))
  (check-it:cached-value generator))

(check-it:def-generator expr-if-tree (&key (n-vars 2) (max-depth 3))
  (make-instance '%expr-if-tree-generator :n-vars n-vars :max-depth max-depth))

(defun %expr-if-to-form (recipe var-symbols)
  (ecase (first recipe)
    (:var (nth (second recipe) var-symbols))
    (:lit (second recipe))
    (:unary (list (%expr-unary-cl-op (second recipe)) (%expr-if-to-form (third recipe) var-symbols)))
    (:nary (list* (second recipe) (mapcar (lambda (r) (%expr-if-to-form r var-symbols)) (cddr recipe))))
    (:if (destructuring-bind (op test-lhs test-rhs then else) (rest recipe)
           (list 'if (list op (%expr-if-to-form test-lhs var-symbols) (%expr-if-to-form test-rhs var-symbols))
                 (%expr-if-to-form then var-symbols)
                 (%expr-if-to-form else var-symbols))))))

(defun %expr-if-eval-plain (recipe values)
  (labels ((ev (r)
             (ecase (first r)
               (:var (nth (second r) values))
               (:lit (coerce (second r) 'single-float))
               (:unary (funcall (fdefinition (%expr-unary-cl-op (second r))) (ev (third r))))
               (:nary (reduce (fdefinition (second r)) (mapcar #'ev (cddr r))))
               (:if (destructuring-bind (op test-lhs test-rhs then else) (rest r)
                      (if (funcall (fdefinition op) (ev test-lhs) (ev test-rhs)) (ev then) (ev else)))))))
    (ev recipe)))

(defun %expr-if-traceable-function (recipe n-vars)
  (let* ((vars (%expr-var-symbols n-vars))
         (body (%expr-if-to-form recipe vars)))
    (eval `(nb:with-tracing ,vars ,body))))

(test trace-if/expr-tree-with-if-eval-graph-matches-direct-call-on-arrays
  "IF/SELECT を含むランダムな式木を TRACE-TO-GRAPH + EVAL-GRAPH した結果は、
配列に直接適用した結果とビット単位で一致する。"
  (is (check-it (generator (tuple (expr-if-tree :n-vars 2 :max-depth 3)
                                   (array-spec :dtypes '(:f32 :f64) :max-rank 3 :max-dim 4)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (recipe spec seed) args
                    (let* ((f (%expr-if-traceable-function recipe 2))
                           (a (make-random-array spec :seed seed))
                           (b (make-random-array spec :seed (1+ seed)))
                           (direct (funcall f a b))
                           (graph (nb:trace-to-graph f (list (nb:array-aval a) (nb:array-aval b))))
                           (traced (nb:eval-graph graph a b)))
                      (%array-bits-equal-p direct traced))))
                :regression-id trace-if/expr-tree-with-if-eval-graph-matches-direct-call-on-arrays
                :regression-file (regression-path "trace-if-expr-tree-matches-eval-graph"))))

(test trace-if/expr-tree-with-if-direct-call-matches-plain-cl-eval-on-scalars
  "スカラーの脚: 直接呼び出しは %EXPR-IF-EVAL-PLAIN（トレースを介さない
ふつうの CL の評価。IF は CL の真偽値で分岐する）と一致する。"
  (is (check-it (generator (tuple (expr-if-tree :n-vars 2 :max-depth 3)
                                   (uniform-real :lo -1.0d0 :hi 1.0d0)
                                   (uniform-real :lo -1.0d0 :hi 1.0d0)))
                (lambda (args)
                  (destructuring-bind (recipe x y) args
                    (let* ((f (%expr-if-traceable-function recipe 2))
                           (vx (coerce x 'single-float))
                           (vy (coerce y 'single-float))
                           (direct (funcall f vx vy))
                           (plain (%expr-if-eval-plain recipe (list vx vy))))
                      (approx= direct plain :dtype :f32))))
                :regression-id trace-if/expr-tree-with-if-direct-call-matches-plain-cl-eval-on-scalars
                :regression-file (regression-path "trace-if-expr-tree-matches-plain-eval"))))

;;;; array-api-test: 配列レベルの公開 API（DOT / RESHAPE / TRANSPOSE /
;;;; BROADCAST-IN-DIM / REDUCE-SUM / REDUCE-MAX / CONVERT / WHERE）の性質
;;;; （issue #32、t2）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; --- eager: 各 API は対応するプリミティブの :EAGER を直接呼んだ結果と
;;; EQUALP になる ---

(defun %primitive-eager-call (name arrays in-avals &rest params)
  (apply (nb::primitive-eager (nb::find-primitive name)) arrays in-avals params))

(test array-api/reshape-eager-matches-primitive
  (let ((a (make-array '(2 3) :element-type 'single-float
                                :initial-contents '((1.0 2.0 3.0) (4.0 5.0 6.0)))))
    (is (equalp (%primitive-eager-call :reshape (list a) (list (nb:array-aval a)) :shape '(6))
                (nb:reshape a '(6))))))

(test array-api/transpose-eager-matches-primitive-with-default-perm
  (let ((a (make-array '(2 3) :element-type 'single-float
                                :initial-contents '((1.0 2.0 3.0) (4.0 5.0 6.0)))))
    (is (equalp (%primitive-eager-call :transpose (list a) (list (nb:array-aval a)) :perm '(1 0))
                (nb:transpose a)))))

(test array-api/transpose-eager-with-explicit-perm
  (let ((a (make-array '(2 3) :element-type 'single-float
                                :initial-contents '((1.0 2.0 3.0) (4.0 5.0 6.0)))))
    (is (equalp (%primitive-eager-call :transpose (list a) (list (nb:array-aval a)) :perm '(0 1))
                (nb:transpose a '(0 1))))))

(test array-api/broadcast-in-dim-eager-matches-primitive
  (let ((a (make-array 2 :element-type 'single-float :initial-contents '(1.0 2.0))))
    (is (equalp (%primitive-eager-call :broadcast-in-dim (list a) (list (nb:array-aval a))
                                        :shape '(3 2) :dims '(1))
                (nb:broadcast-in-dim a '(3 2) '(1))))))

(test array-api/reduce-sum-eager-with-default-axes-reduces-all
  (let ((a (make-array '(2 3) :element-type 'single-float
                                :initial-contents '((1.0 2.0 3.0) (4.0 5.0 6.0)))))
    (is (equalp (%primitive-eager-call :reduce-sum (list a) (list (nb:array-aval a)) :axes '(0 1))
                (nb:reduce-sum a)))))

(test array-api/reduce-sum-eager-with-explicit-axes
  (let ((a (make-array '(2 3) :element-type 'single-float
                                :initial-contents '((1.0 2.0 3.0) (4.0 5.0 6.0)))))
    (is (equalp (%primitive-eager-call :reduce-sum (list a) (list (nb:array-aval a)) :axes '(1))
                (nb:reduce-sum a :axes '(1))))))

(test array-api/reduce-max-eager-with-default-axes-reduces-all
  (let ((a (make-array '(2 3) :element-type 'single-float
                                :initial-contents '((1.0 2.0 3.0) (4.0 5.0 6.0)))))
    (is (equalp (%primitive-eager-call :reduce-max (list a) (list (nb:array-aval a)) :axes '(0 1))
                (nb:reduce-max a)))))

(test array-api/reduce-sum-eager-with-explicit-empty-axes-is-identity
  "AXES を明示的に空リストで渡すと reduce しない（X をそのまま返す）。
省略時（デフォルトの全軸）と区別できなければならない。"
  (let ((a (make-array '(2 3) :element-type 'single-float
                                :initial-contents '((1.0 1.0 1.0) (1.0 1.0 1.0)))))
    (is (equalp a (nb:reduce-sum a :axes '())))
    (is (not (equalp a (nb:reduce-sum a))))))

(test array-api/reduce-max-eager-with-explicit-empty-axes-is-identity
  (let ((a (make-array '(2 3) :element-type 'single-float
                                :initial-contents '((1.0 2.0 3.0) (4.0 5.0 6.0)))))
    (is (equalp a (nb:reduce-max a :axes '())))
    (is (not (equalp a (nb:reduce-max a))))))

(test array-api/traced-reduce-sum-explicit-empty-axes-adds-no-eqn
  "トレース時も同じ規約: AXES が明示的に空なら EQN を足さず、そのまま
入力の VAR を返す（0 個の EQN のグラフ）。"
  (let* ((f (nb:with-tracing (x) (nb:reduce-sum x :axes '())))
         (graph (nb:trace-to-graph f (list (nb:make-aval '(2 3) :f32)))))
    (is (null (nb:graph-eqns graph)))
    (is (eq (first (nb:graph-invars graph)) (first (nb:graph-outvars graph))))))

(test array-api/traced-reduce-max-explicit-empty-axes-adds-no-eqn
  (let* ((f (nb:with-tracing (x) (nb:reduce-max x :axes '())))
         (graph (nb:trace-to-graph f (list (nb:make-aval '(2 3) :f32)))))
    (is (null (nb:graph-eqns graph)))
    (is (eq (first (nb:graph-invars graph)) (first (nb:graph-outvars graph))))))

(test array-api/convert-eager-matches-primitive
  (let ((a (make-array 2 :element-type 'single-float :initial-contents '(1.0 2.0))))
    (is (equalp (%primitive-eager-call :convert (list a) (list (nb:array-aval a)) :dtype :f64)
                (nb:convert a :f64)))))

(test array-api/dot-eager-2d-matches-primitive
  (let ((a (make-array '(2 3) :element-type 'single-float
                                :initial-contents '((1.0 2.0 3.0) (4.0 5.0 6.0))))
        (b (make-array '(3 4) :element-type 'single-float
                                :initial-contents '((1.0 0.0 0.0 1.0) (0.0 1.0 0.0 1.0) (0.0 0.0 1.0 1.0)))))
    (is (equalp (%primitive-eager-call :dot-general (list a b) (list (nb:array-aval a) (nb:array-aval b))
                                        :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '())
                (nb:dot a b)))))

(test array-api/dot-eager-1d-1d-contracts-to-scalar
  (let ((a (make-array 3 :element-type 'single-float :initial-contents '(1.0 2.0 3.0)))
        (b (make-array 3 :element-type 'single-float :initial-contents '(4.0 5.0 6.0))))
    (is (equalp (%primitive-eager-call :dot-general (list a b) (list (nb:array-aval a) (nb:array-aval b))
                                        :lhs-contracting '(0) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '())
                (nb:dot a b)))
    (is (= 32.0 (aref (nb:dot a b))))))

(test array-api/dot-eager-2d-1d-contracts-lhs-last-axis
  "2次元・1次元の DOT は A の最後の軸（rank-1）を縮約する（オフバイワン
対策: (1- rank) が 0 に化けると、代わりに1番目の軸を縮約してしまう）。"
  (let ((a (make-array '(2 3) :element-type 'single-float
                                :initial-contents '((1.0 0.0 0.0) (0.0 1.0 0.0))))
        (b (make-array 3 :element-type 'single-float :initial-contents '(10.0 20.0 30.0))))
    (is (equalp #(10.0 20.0) (nb:dot a b)))))

(test array-api/dot-rank0-signals-tracing-error
  (let ((a (make-array '() :element-type 'single-float :initial-element 1.0))
        (b (make-array '() :element-type 'single-float :initial-element 2.0)))
    (signals nb:tracing-error (nb:dot a b))))

(test array-api/dot-rank0-second-operand-signals-tracing-error
  "A が rank 1 以上でも、B が rank 0 なら TRACING-ERROR になる（A だけを
チェックして B を見落とす変異対策）。"
  (let ((a (make-array 3 :element-type 'single-float :initial-contents '(1.0 2.0 3.0)))
        (b (make-array '() :element-type 'single-float :initial-element 2.0)))
    (signals nb:tracing-error (nb:dot a b))))

(test array-api/where-eager-matches-select
  (let* ((pred (make-array 2 :element-type 'bit :initial-contents '(1 0)))
         (a (make-array 2 :element-type 'single-float :initial-contents '(1.0 2.0)))
         (b (make-array 2 :element-type 'single-float :initial-contents '(10.0 20.0))))
    (is (equalp (%primitive-eager-call :select (list pred a b)
                                        (list (nb:array-aval pred) (nb:array-aval a) (nb:array-aval b)))
                (nb:where pred a b)))))

(test array-api/where-eager-array-pred-with-tracer-branch-lifts-pred
  "WHERE の PRED が eager な bit 配列でも、A・B の少なくとも一方がトレーサ
なら PRED をリフトしてトレースする（回帰: 直接 %EAGER-SELECT-ARRAY に
渡すとトレーサが配列演算に落ちて SIMPLE-TYPE-ERROR になる）。"
  (let* ((pred (make-array 2 :element-type 'bit :initial-contents '(1 0)))
         (f (nb:with-tracing (x) (nb:where pred x 0.0)))
         (graph (nb:trace-to-graph f (list (nb:make-aval '(2) :f32))))
         (x (make-array 2 :element-type 'single-float :initial-contents '(3.0 4.0))))
    (is (find :select (nb:graph-eqns graph) :key (lambda (e) (nb:primitive-name (nb:eqn-prim e)))))
    (is (equalp #(3.0 0.0) (funcall f x)))
    (is (equalp #(3.0 0.0) (nb:eval-graph graph x)))))

(test array-api/where-eager-with-number-branch
  "片方の分岐が数値（実数）なら、もう一方の分岐（配列）の dtype・PRED の
shape にリフトしてから SELECT する。両方が数値だと dtype を決められず
TRACING-ERROR になる（%T-SELECT の仕様）。"
  (let* ((pred (make-array 2 :element-type 'bit :initial-contents '(1 0)))
         (a (make-array 2 :element-type 'single-float :initial-contents '(1.0 2.0))))
    (is (equalp #(1.0 0.0) (nb:where pred a 0.0)))
    (signals nb:tracing-error (nb:where pred 1.0 0.0))))

;;; --- traced: グラフの eqn とパラメタを goldens で確かめる ---

(test array-api/traced-dot-golden
  (is (string= "(graph
 (:in (%0 f32 (2 3)) (%1 f32 (3 4)))
 (:const)
 (:eqns
  (%2 f32 (2 4) := dot-general (:lhs-contracting (1) :rhs-contracting (0) :lhs-batch () :rhs-batch ()) %0 %1))
 (:out %2))"
               (%print-graph-string (nb:with-tracing (a b) (nb:dot a b))
                                    (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(3 4) :f32))))))

(test array-api/traced-reshape-golden
  (is (string= "(graph
 (:in (%0 f32 (2 3)))
 (:const)
 (:eqns
  (%1 f32 (6) := reshape (:shape (6)) %0))
 (:out %1))"
               (%print-graph-string (nb:with-tracing (x) (nb:reshape x '(6)))
                                    (list (nb:make-aval '(2 3) :f32))))))

(test array-api/traced-transpose-default-perm-golden
  "デフォルトの perm は軸を逆順にする（rank 2 なら (1 0)）。"
  (is (string= "(graph
 (:in (%0 f32 (2 3)))
 (:const)
 (:eqns
  (%1 f32 (3 2) := transpose (:perm (1 0)) %0))
 (:out %1))"
               (%print-graph-string (nb:with-tracing (x) (nb:transpose x))
                                    (list (nb:make-aval '(2 3) :f32))))))

(test array-api/traced-broadcast-in-dim-golden
  (is (string= "(graph
 (:in (%0 f32 (2)))
 (:const)
 (:eqns
  (%1 f32 (3 2) := broadcast-in-dim (:shape (3 2) :dims (1)) %0))
 (:out %1))"
               (%print-graph-string (nb:with-tracing (x) (nb:broadcast-in-dim x '(3 2) '(1)))
                                    (list (nb:make-aval '(2) :f32))))))

(test array-api/traced-reduce-sum-default-axes-golden
  "デフォルトの axes は全軸（rank 2 なら (0 1)）。"
  (is (string= "(graph
 (:in (%0 f32 (2 3)))
 (:const)
 (:eqns
  (%1 f32 () := reduce-sum (:axes (0 1)) %0))
 (:out %1))"
               (%print-graph-string (nb:with-tracing (x) (nb:reduce-sum x))
                                    (list (nb:make-aval '(2 3) :f32))))))

(test array-api/traced-reduce-max-explicit-axes-golden
  (is (string= "(graph
 (:in (%0 f32 (2 3)))
 (:const)
 (:eqns
  (%1 f32 (2) := reduce-max (:axes (1)) %0))
 (:out %1))"
               (%print-graph-string (nb:with-tracing (x) (nb:reduce-max x :axes '(1)))
                                    (list (nb:make-aval '(2 3) :f32))))))

(test array-api/traced-convert-golden
  (is (string= "(graph
 (:in (%0 f32 (2)))
 (:const)
 (:eqns
  (%1 f64 (2) := convert (:dtype :f64) %0))
 (:out %1))"
               (%print-graph-string (nb:with-tracing (x) (nb:convert x :f64))
                                    (list (nb:make-aval '(2) :f32))))))

(test array-api/traced-where-golden
  (is (string= "(graph
 (:in (%0 i1 (2)) (%1 f32 (2)) (%2 f32 (2)))
 (:const)
 (:eqns
  (%3 f32 (2) := select () %0 %1 %2))
 (:out %3))"
               (%print-graph-string (nb:with-tracing (p a b) (nb:where p a b))
                                    (list (nb:make-aval '(2) :i1) (nb:make-aval '(2) :f32) (nb:make-aval '(2) :f32))))))

;;; --- 性質: transpose を2回、reshape を往復、reduce-sum のデフォルト ---

(test array-api/transpose-twice-default-perm-is-identity-eager
  (let ((spec (make-array-spec '(2 3) :f32)))
    (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                  (lambda (seed)
                    (let* ((a (make-random-array spec :seed seed))
                           (twice (nb:transpose (nb:transpose a))))
                      (equalp a twice)))))))

(test array-api/transpose-twice-default-perm-is-identity-traced
  (let* ((f (nb:with-tracing (x) (nb:transpose (nb:transpose x))))
         (graph (nb:trace-to-graph f (list (nb:make-aval '(2 3) :f32))))
         (a (make-random-array (make-array-spec '(2 3) :f32) :seed 7)))
    (is (equalp a (nb:eval-graph graph a)))))

(test array-api/reshape-to-flat-then-back-is-identity
  (let* ((spec (make-array-spec '(2 3) :f32))
         (a (make-random-array spec :seed 11))
         (f (nb:with-tracing (x) (nb:reshape (nb:reshape x '(6)) '(2 3))))
         (graph (nb:trace-to-graph f (list (nb:array-aval a)))))
    (is (equalp a (funcall f a)))
    (is (equalp a (nb:eval-graph graph a)))))

(test array-api/reduce-sum-default-matches-reduce-over-all-axes
  (let* ((spec (make-array-spec '(2 3) :f32))
         (a (make-random-array spec :seed 13)))
    (is (equalp (nb:reduce-sum a) (nb:reduce-sum a :axes '(0 1))))))

;;; --- 性質: rank 0 のトレーサ/配列 x rank2 のトレーサ/配列、両方の順序で
;;; ブロードキャストされる（issue #32 契約のピットフォール(7)） ---

(test array-api/rank0-tracer-broadcasts-against-higher-rank-both-orders
  (let* ((f-left (nb:with-tracing (s x) (+ (nb:reduce-sum s) x)))
         (f-right (nb:with-tracing (s x) (+ x (nb:reduce-sum s))))
         (avals (list (nb:make-aval '(2) :f32) (nb:make-aval '(2 3) :f32)))
         (graph-left (nb:trace-to-graph f-left avals))
         (graph-right (nb:trace-to-graph f-right avals))
         (s (make-random-array (make-array-spec '(2) :f32) :seed 17))
         (x (make-random-array (make-array-spec '(2 3) :f32) :seed 19)))
    (is (equalp (funcall f-left s x) (nb:eval-graph graph-left s x)))
    (is (equalp (funcall f-right s x) (nb:eval-graph graph-right s x)))))

(test array-api/rank0-array-broadcasts-against-higher-rank-both-orders
  (let* ((s (nb:reduce-sum (make-random-array (make-array-spec '(2) :f32) :seed 23)))
         (x (make-random-array (make-array-spec '(2 3) :f32) :seed 29))
         (broadcast-s (nb:broadcast-in-dim s '(2 3) '()))
         (expected (%primitive-eager-call :add (list broadcast-s x)
                                           (list (nb:array-aval broadcast-s) (nb:array-aval x)))))
    (is (equalp expected (nb::%t-add s x)))
    (is (equalp expected (nb::%t-add x s)))))

;;; --- 最終 PBT（#32 の criterion）: 算術・IF/SELECT・配列 API（reshape,
;;; transpose, 1軸だけの reduce, dot、K ≤ 4）を混ぜた式の
;;; TRACE-TO-GRAPH + EVAL-GRAPH == 直接呼び出し ---
;;;
;;; 形状は M・K・N（すべて 1〜4）で決まる: A・B は (M K)、W は (K N)。
;;;   X = (IF (> A 0.0) (+ A B) (- A B))     ; 算術 + IF/SELECT（要素ごとの
;;;                                            PRED なので分岐と shape が
;;;                                            一致する。rank 0 の PRED は
;;;                                            trace-if-test で別に確かめる）
;;;   Y = (DOT X W)                          ; (M N)、縮約軸は K（≤ 4）
;;;   Z = (TRANSPOSE Y)                      ; (N M)
;;;   FLAT = (RESHAPE Z (N*M))               ; reshape で flatten
;;;   FLAT2 = (RESHAPE FLAT (N M))           ; reshape で戻す
;;;   R = (REDUCE-SUM FLAT2 :AXES (1))       ; 1軸だけの reduce（全軸ではない）
;;; M・K・N は WITH-TRACING のコードに直接埋め込む（形状はトレース時に
;;; 固定されていなければならない）ので、式そのものを EVAL で組み立てる
;;; （%EXPR-IF-TRACEABLE-FUNCTION 等と同じ考え方）。

(defun %mixed-pbt-form (m k n)
  "M・K・N（配列 A/B の shape (M K)、W の shape (K N)）から、算術・
IF/SELECT・DOT/TRANSPOSE/RESHAPE/REDUCE-SUM を混ぜた WITH-TRACING の
フォームを組み立てる。"
  `(nb:with-tracing (a b w)
     (let* ((x (if (> a 0.0) (+ a b) (- a b)))
            (y (nb:dot x w))
            (z (nb:transpose y))
            (flat (nb:reshape z (list ,(* n m))))
            (flat2 (nb:reshape flat (list ,n ,m)))
            (r (nb:reduce-sum flat2 :axes (list 1))))
       r)))

(test array-api/mixed-arithmetic-if-select-and-array-api-eval-graph-matches-direct-call
  "算術（+/-）・IF/SELECT・DOT/TRANSPOSE/RESHAPE/REDUCE-SUM（1軸）を混ぜた
式の TRACE-TO-GRAPH + EVAL-GRAPH は、TRACEABLE-FUNCTION を配列に直接
適用した結果とビット単位で一致する（#32 の最終基準）。"
  (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 4)
                                   (uniform-integer :lo 1 :hi 4)
                                   (uniform-integer :lo 1 :hi 4)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (m k n seed) args
                    (let* ((f (eval (%mixed-pbt-form m k n)))
                           (spec-a (make-array-spec (list m k) :f32))
                           (spec-w (make-array-spec (list k n) :f32))
                           (a (make-random-array spec-a :seed seed))
                           (b (make-random-array spec-a :seed (+ seed 1)))
                           (w (make-random-array spec-w :seed (+ seed 2)))
                           (direct (funcall f a b w))
                           (graph (nb:trace-to-graph f (list (nb:array-aval a) (nb:array-aval b) (nb:array-aval w))))
                           (traced (nb:eval-graph graph a b w)))
                      (%array-bits-equal-p direct traced))))
                :regression-id array-api/mixed-arithmetic-if-select-and-array-api-eval-graph-matches-direct-call
                :regression-file (regression-path "array-api-mixed-matches-eval-graph"))))

;;; --- issue #74: DOT で配列とトレーサを混ぜる、rank 0 の PRED の
;;; ブロードキャスト、Lisp のブール値の PRED ---
;;;
;;; dtype は :f32 / :f64 だけを使う。bf16 / f16 は生の (UNSIGNED-BYTE 16)
;;; 配列から ARRAY-AVAL が dtype を推論できず、トレース中に定数として
;;; リフトできないため（性質そのものは dtype に依らない）。

(defparameter *array-api-float-dtypes* '(:f32 :f64))

(test array-api/dot-mixing-array-and-tracer-matches-array-dot
  "DOT の片方が定数の配列、もう片方がトレーサでも（両方の順序で）
トレースでき、EVAL-GRAPH の結果は配列どうしの DOT とビット単位で一致
する（issue #74）。"
  (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 4)
                                   (uniform-integer :lo 1 :hi 4)
                                   (uniform-integer :lo 1 :hi 4)
                                   (uniform-integer :lo 0 :hi 1)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (m k n dtype-index seed) args
                    (let* ((dtype (nth dtype-index *array-api-float-dtypes*))
                           (c (make-random-array (make-array-spec (list m k) dtype) :seed seed))
                           (x (make-random-array (make-array-spec (list k n) dtype) :seed (+ seed 1)))
                           (y (make-random-array (make-array-spec (list n m) dtype) :seed (+ seed 2)))
                           (const-lhs (nb:with-tracing (x) (nb:dot c x)))
                           (const-rhs (nb:with-tracing (y) (nb:dot y c))))
                      (and (equalp (nb:dot c x)
                                   (nb:eval-graph (nb:trace-to-graph const-lhs (list (nb:array-aval x))) x))
                           (equalp (nb:dot y c)
                                   (nb:eval-graph (nb:trace-to-graph const-rhs (list (nb:array-aval y))) y))))))
                :regression-id array-api/dot-mixing-array-and-tracer-matches-array-dot
                :regression-file (regression-path "array-api-dot-mixing-array-and-tracer"))))

(test array-api/dot-mixing-rank0-operand-signals-tracing-error
  "配列とトレーサを混ぜた DOT でも、どちらかが rank 0 なら TRACING-ERROR
になる（配列どうし・トレーサどうしと同じ規約）。"
  (let ((scalar (make-array '() :element-type 'single-float :initial-element 2.0))
        (vector (make-array 3 :element-type 'single-float :initial-contents '(1.0 2.0 3.0))))
    (signals nb:tracing-error
      (nb:trace-to-graph (nb:with-tracing (x) (nb:dot scalar x)) (list (nb:make-aval '(3) :f32))))
    (signals nb:tracing-error
      (nb:trace-to-graph (nb:with-tracing (x) (nb:dot x scalar)) (list (nb:make-aval '(3) :f32))))
    (signals nb:tracing-error
      (nb:trace-to-graph (nb:with-tracing (x) (nb:dot vector x)) (list (nb:make-aval '() :f32))))))

(test array-api/where-rank0-pred-broadcasts-to-branch-shape
  "PRED が rank 0 で A・B が rank 1 以上なら、PRED を A・B の shape に
ブロードキャストする（JAX の where と同じ）。結果は PRED のビットに応じて
A か B のどちらか全体になる。eager・トレーサの PRED・eager な PRED と
トレーサの分岐、のどれでも同じ（issue #74）。"
  (is (check-it (generator (tuple (array-spec :dtypes *array-api-float-dtypes* :max-rank 3)
                                   (uniform-integer :lo 0 :hi 1)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec bit seed) args
                    (let* ((pred (make-array '() :element-type 'bit :initial-element bit))
                           (a (make-random-array spec :seed seed))
                           (b (make-random-array spec :seed (+ seed 1)))
                           (expected (if (= bit 1) a b))
                           (traced-pred (nb:trace-to-graph (nb:with-tracing (p x y) (nb:where p x y))
                                                           (list (nb:make-aval '() :i1) (nb:array-aval a) (nb:array-aval b))))
                           (traced-branches (nb:trace-to-graph (nb:with-tracing (x y) (nb:where pred x y))
                                                               (list (nb:array-aval a) (nb:array-aval b)))))
                      (and (equalp expected (nb:where pred a b))
                           (equalp expected (nb:eval-graph traced-pred pred a b))
                           (equalp expected (nb:eval-graph traced-branches a b))))))
                :regression-id array-api/where-rank0-pred-broadcasts-to-branch-shape
                :regression-file (regression-path "array-api-where-rank0-pred-broadcasts"))))

(test array-api/where-rank0-pred-with-number-branch-uses-array-branch-shape
  "PRED が rank 0 で片方の分岐が数値なら、shape はもう一方（配列／
トレーサ）の分岐から決まる。"
  (let* ((pred (make-array '() :element-type 'bit :initial-element 0))
         (a (make-array 3 :element-type 'single-float :initial-contents '(1.0 2.0 3.0)))
         (f (nb:with-tracing (p x) (nb:where p x 0.0)))
         (graph (nb:trace-to-graph f (list (nb:make-aval '() :i1) (nb:array-aval a)))))
    (is (equalp #(0.0 0.0 0.0) (nb:where pred a 0.0)))
    (is (equalp #(0.0 0.0 0.0) (nb:eval-graph graph pred a)))))

(test array-api/where-lisp-boolean-pred-chooses-branch-statically
  "PRED が Lisp のブール値（T / NIL）なら、ふつうの IF と同じく片方の
分岐を静的に選ぶ。SELECT の EQN を足さず、配列・トレーサの分岐は
そのまま返る（issue #74）。"
  (is (check-it (generator (tuple (array-spec :dtypes *array-api-float-dtypes* :max-rank 3)
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((a (make-random-array spec :seed seed))
                           (b (make-random-array spec :seed (+ seed 1)))
                           (avals (list (nb:array-aval a) (nb:array-aval b)))
                           (graph-t (nb:trace-to-graph (nb:with-tracing (x y) (nb:where t x y)) avals))
                           (graph-nil (nb:trace-to-graph (nb:with-tracing (x y) (nb:where nil x y)) avals)))
                      (and (eq a (nb:where t a b))
                           (eq b (nb:where nil a b))
                           (null (nb:graph-eqns graph-t))
                           (eq (first (nb:graph-invars graph-t)) (first (nb:graph-outvars graph-t)))
                           (null (nb:graph-eqns graph-nil))
                           (eq (second (nb:graph-invars graph-nil)) (first (nb:graph-outvars graph-nil)))))))
                :regression-id array-api/where-lisp-boolean-pred-chooses-branch-statically
                :regression-file (regression-path "array-api-where-lisp-boolean-pred"))))

(test array-api/where-lisp-boolean-pred-broadcasts-chosen-branch-like-rank0-pred
  "PRED が T / NIL のときの結果の shape・dtype は、同じ真偽の rank 0 の
PRED を渡したときと同じ: 選んだ分岐が数値・rank 0 なら、もう一方の分岐の
shape・dtype に合わせてブロードキャストする。"
  (let* ((a (make-array 3 :element-type 'single-float :initial-contents '(1.0 2.0 3.0)))
         (s (make-array '() :element-type 'single-float :initial-element 5.0))
         (f (nb:with-tracing (x) (nb:where nil x 7.0)))
         (graph (nb:trace-to-graph f (list (nb:array-aval a)))))
    (is (equalp #(0.0 0.0 0.0) (nb:where t 0.0 a)))
    (is (equalp #(5.0 5.0 5.0) (nb:where nil a s)))
    (is (equalp #(7.0 7.0 7.0) (nb:eval-graph graph a)))
    (is (equalp #(7.0 7.0 7.0) (funcall f a)))))

(test array-api/where-lisp-boolean-pred-keeps-branch-checks
  "PRED が T / NIL でも、分岐の検査（両方数値・:I1 の分岐は TRACING-ERROR）
は rank 0 の PRED と同じ。選ばれない側の分岐も検査する。"
  (let ((a (make-array 2 :element-type 'single-float :initial-contents '(1.0 2.0)))
        (bits (make-array 2 :element-type 'bit :initial-contents '(1 0))))
    (signals nb:tracing-error (nb:where t 1.0 2.0))
    (signals nb:tracing-error (nb:where t a bits))
    (signals nb:tracing-error (nb:where nil bits a))))

(test array-api/where-non-boolean-plain-pred-signals-tracing-error
  "PRED が配列・トレーサ・T・NIL のどれでもない（数値など）なら
NO-APPLICABLE-METHOD ではなく TRACING-ERROR になる。"
  (let ((a (make-array 2 :element-type 'single-float :initial-contents '(1.0 2.0))))
    (signals nb:tracing-error (nb:where 1 a a))
    (signals nb:tracing-error (nb:where :yes a a))))

(test array-api/where-lisp-boolean-pred-checks-unchosen-branch-shape-and-dtype
  "PRED が T / NIL でも、選ばれない側の分岐と shape・dtype が食い違えば
（rank 0 の PRED を渡したときと同じく）エラーになる。真偽値によって
エラーになったりならなかったりしない。"
  (let ((a3 (make-array 3 :element-type 'single-float :initial-element 1.0))
        (a4 (make-array 4 :element-type 'single-float :initial-element 2.0))
        (d3 (make-array 3 :element-type 'double-float :initial-element 3d0)))
    (signals nb:primitive-error (nb:where t a3 a4))
    (signals nb:primitive-error (nb:where nil a3 a4))
    (signals nb:dtype-mismatch
      (nb:trace-to-graph (nb:with-tracing (x) (nb:where t x d3)) (list (nb:make-aval '(3) :f32))))
    (signals nb:dtype-mismatch
      (nb:trace-to-graph (nb:with-tracing (x) (nb:where nil x d3)) (list (nb:make-aval '(3) :f32))))))

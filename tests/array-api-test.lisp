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

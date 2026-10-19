;;;; cond* / while-loop / scan の carry・xs・ys・operands の PyTree（issue #201、small）。
;;;;
;;;; 守らせる性質:
;;;; - while-loop: 入れ子の木（リスト・ベクタ・登録した型）の carry で、eager もトレースして
;;;;   eval-graph した結果も、本体を Lisp のループで回した参照実装と一致し、結果の treedef は
;;;;   init と同じ。
;;;; - cond*: 木の operands と木の戻り値で、トレースした結果が選ばれた枝を直接呼んだ結果と一致する。
;;;; - scan: 木の init / xs / ys で、結果（carry と ys）の値と treedef が Lisp のループによる
;;;;   参照実装と一致する（reverse を含む）。
;;;; - 本体が返す木の形（treedef）が入力と違えば、各構文の既存のコンディションで報告する。
;;;; - 木の carry の scan を grad（f64 の中心差分と一致）と vmap（要素ごとの参照実装と一致）に通せる。
;;;; - 既存のリストの呼び出しは、既存のテスト（while-loop-test / cond-test / scan-test）が守る。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defparameter *ctl-scale* (nb:with-tracing (x) (* 0.9d0 x)))
(defparameter *ctl-step* (nb:with-tracing (c s) (tanh (+ c s))))
(defparameter *ctl-inc* (nb:with-tracing (x) (+ x 1.0d0)))
(defparameter *ctl-lt* (nb:with-tracing (a b) (< a b)))

(defun %ctl-scale-tree (tree)
  (nb:tree-map (lambda (x) (funcall *ctl-scale* x)) tree))

(defun %ctl-scalar (x) (nb::%scalar-array x :f64))

(defun %ctl-traced-call (fn args)
  "FN を ARGS（木のリスト）で、トレースして eval-graph で評価した結果（FN の戻り値を多値で）。
葉は ARGS を flatten したもの（jit と同じ手順で、IREE を使わない）。"
  (multiple-value-bind (leaves in-treedef) (nb::%flatten-arguments args)
    (multiple-value-bind (flat out-treedef) (nb::%flatten-function fn in-treedef)
      (let* ((tf (nb::%make-traceable-function (loop for nil in leaves collect (gensym "L"))
                                               (lambda (&rest xs) (apply flat xs))))
             (graph (nb::trace-to-graph tf (mapcar #'nb::array-aval leaves)))
             (outs (multiple-value-list (apply #'nb:eval-graph graph leaves))))
        (nb::%unflatten-results (funcall out-treedef) outs)))))

(defun %ctl-same-tree-p (actual expected)
  "ACTUAL と EXPECTED が同じ treedef で、葉が f64 の許容誤差内で一致する。"
  (and (nb:treedef= (nb:tree-structure actual) (nb:tree-structure expected))
       (every (lambda (a e) (allclose a e :dtype :f64))
              (nb:tree-leaves actual) (nb:tree-leaves expected))))

(defun %ctl-has-leaves-p (tree) (plusp (length (nb:tree-leaves tree))))

(defmacro def-pytree-control-property (name docstring (&rest vars) gen &body body)
  `(test ,name ,docstring
     (is (check-it ,gen
                   (lambda (value) (destructuring-bind ,vars value ,@body))
                   :regression-id ,name
                   :regression-file (regression-path
                                     ,(format nil "pytree-control-~(~A~)"
                                              (substitute #\- #\/ (subseq (string name)
                                                                          (length "pytree-control/"))))))
         ,(format nil "~A の性質が成り立たなかった" name))))

;;; ---- while-loop ----

(defun %ctl-wl-carry (spec limit)
  "carry: (pytree-test-pair #(counter limit) 木 7)。入れ子の節を混ぜた木。"
  (make-pytree-test-pair (vector (%ctl-scalar 0d0) (%ctl-scalar (float limit 1d0)))
                         (pytree-array-tree spec :f64 :max-leaves 5)
                         7))

(defun %ctl-wl-cond (c)
  (let ((state (pytree-test-pair-left c)))
    (funcall *ctl-lt* (svref state 0) (svref state 1))))

(defun %ctl-wl-body (c)
  (let ((state (pytree-test-pair-left c)))
    (make-pytree-test-pair (vector (funcall *ctl-inc* (svref state 0)) (svref state 1))
                           (%ctl-scale-tree (pytree-test-pair-right c))
                           7)))

(defun %ctl-wl-reference (c limit)
  (dotimes (i limit c) (setf c (%ctl-wl-body c))))

(def-pytree-control-property pytree-control/while-loop-tree-carry-matches-reference
  "入れ子の木の carry の while-loop は、eager でもトレースしても、本体を limit 回(0〜4)回した
参照実装と同じ treedef・同じ値になる。"
  (spec limit) (generator (tuple (pytree-spec) (uniform-integer :lo 0 :hi 4)))
  (let* ((c (%ctl-wl-carry spec limit))
         (expected (%ctl-wl-reference c limit)))
    (and (%ctl-same-tree-p (nb:while-loop #'%ctl-wl-cond #'%ctl-wl-body c) expected)
         (%ctl-same-tree-p
          (%ctl-traced-call (lambda (c) (nb:while-loop #'%ctl-wl-cond #'%ctl-wl-body c)) (list c))
          expected))))

(test pytree-control/while-loop-accepts-a-bare-array-and-a-vector
  "carry は配列そのもの（葉1つの木）や simple-vector でもよく、結果は同じ形で返る。"
  (let* ((x (%ctl-scalar 0d0))
         (cond-fn (lambda (c) (funcall *ctl-lt* c 3d0)))
         (result (nb:while-loop cond-fn (lambda (c) (funcall *ctl-inc* c)) x)))
    (is (arrayp result))
    (is (= 3d0 (aref result))))
  (let ((result (nb:while-loop (lambda (c) (funcall *ctl-lt* (svref c 0) 2d0))
                               (lambda (c) (vector (funcall *ctl-inc* (svref c 0))))
                               (vector (%ctl-scalar 0d0)))))
    (is (simple-vector-p result))
    (is (= 2d0 (aref (svref result 0))))))

(test pytree-control/while-loop-rejects-treedef-mismatch-and-leafless-init
  "本体が返す木の形が init と違えば（葉の個数と aval が同じでも）WHILE-LOOP-CARRY-MISMATCH。
葉の無い init は WHILE-LOOP-ARGUMENT-ERROR。"
  (let ((init (list (%ctl-scalar 0d0) (%ctl-scalar 1d0)))
        (cond-fn (lambda (c) (funcall *ctl-lt* (if (consp c) (first c) c) 5d0))))
    ;; どの本体も先頭の葉を増やすので、treedef の検査が無ければ 5 まで回って結果が食い違う
    ;; （止まらないループにはならない）
    (signals nb:while-loop-carry-mismatch
      (nb:while-loop cond-fn (lambda (c) (list (list (funcall *ctl-inc* (first c)) (second c)))) init))
    (signals nb:while-loop-carry-mismatch
      (nb:while-loop cond-fn (lambda (c) (list (funcall *ctl-inc* (first c)) (list (second c)))) init))
    (signals nb:while-loop-carry-mismatch
      (nb:while-loop cond-fn
                     (lambda (c) (coerce (list (funcall *ctl-inc* (first c)) (second c)) 'simple-vector))
                     init))
    (signals nb:while-loop-argument-error (nb:while-loop cond-fn #'identity '()))
    (signals nb:while-loop-argument-error (nb:while-loop cond-fn #'identity (list nil)))))

(test pytree-control/while-loop-carry-mismatch-message-shows-both-treedefs
  "木の形の不一致のメッセージには、init と本体の戻り値の木の形（treedef）が両方出る。"
  (let ((init (list (%ctl-scalar 0d0) (%ctl-scalar 1d0))))
    (handler-case
        (progn (nb:while-loop (lambda (c) (funcall *ctl-lt* (first c) 5d0))
                              (lambda (c) (list (list (funcall *ctl-inc* (first c)) (second c))))
                              init)
               (fail "木の形の不一致が検出されなかった"))
      (nb:while-loop-carry-mismatch (condition)
        (let ((text (princ-to-string condition)))
          (is (search "(* *)" text))
          (is (search "((* *))" text)))))))

;;; ---- cond* ----

(defparameter *ctl-then*
  (nb::%make-traceable-function
   (list 'a 'b) (lambda (a b) (list (%ctl-scale-tree a) b))))

(defparameter *ctl-else*
  (nb::%make-traceable-function
   (list 'a 'b) (lambda (a b) (list a (%ctl-scale-tree (%ctl-scale-tree b))))))

(def-pytree-control-property pytree-control/cond-tree-operands-match-the-chosen-branch
  "木の operands と木の戻り値の cond* は、トレースして評価した結果が、選ばれた枝を直接呼んだ
結果と同じ treedef・同じ値になる（pred は真・偽の両方）。"
  (spec-a spec-b pred) (generator (tuple (pytree-spec) (pytree-spec) (uniform-integer :lo 0 :hi 1)))
  (let ((a (pytree-array-tree spec-a :f64 :max-leaves 3))
        (b (pytree-array-tree spec-b :f64 :max-leaves 3))
        (p (%ctl-scalar (if (= pred 1) 0d0 1d0))))
    (or (not (or (%ctl-has-leaves-p a) (%ctl-has-leaves-p b)))
        (%ctl-same-tree-p
         (%ctl-traced-call
          (lambda (p a b)
            (nb:cond* (funcall *ctl-lt* p 0.5d0) *ctl-then* *ctl-else* a b))
          (list p a b))
         (funcall (if (= pred 1) *ctl-then* *ctl-else*) a b)))))

(test pytree-control/cond-rejects-branches-with-different-treedefs
  "両枝の戻り値の木の形が違えば（葉の個数と aval が同じでも）COND-ERROR。"
  (let ((then (nb::%make-traceable-function (list 'a) (lambda (a) (list a a))))
        (else (nb::%make-traceable-function (list 'a) (lambda (a) (list (list a) a))))
        (leafless (nb::%make-traceable-function (list 'a) (lambda (a) (declare (ignore a)) '()))))
    (signals nb:cond-error
      (%ctl-traced-call (lambda (p a) (nb:cond* (funcall *ctl-lt* p 0.5d0) then else a))
                        (list (%ctl-scalar 0d0) (%ctl-scalar 1d0))))
    (signals nb:cond-error
      (%ctl-traced-call (lambda (p a) (nb:cond* (funcall *ctl-lt* p 0.5d0) leafless leafless a))
                        (list (%ctl-scalar 0d0) (%ctl-scalar 1d0))))))

(test pytree-control/cond-rejects-a-branch-result-that-is-not-a-pytree
  "枝の戻り値が PyTree として不正（真リストでないコンス）なら COND-ERROR。"
  (let ((bad (nb::%make-traceable-function (list 'a) (lambda (a) (cons a 1)))))
    (signals nb:cond-error
      (%ctl-traced-call (lambda (p a) (nb:cond* (funcall *ctl-lt* p 0.5d0) bad bad a))
                        (list (%ctl-scalar 0d0) (%ctl-scalar 1d0))))))

(test pytree-control/cond-propagates-a-pytree-error-raised-inside-a-branch-body
  "枝の本体の中で起きた PYTREE-ERROR（ユーザーのコードの tree-map の形の不一致など）は、
COND-ERROR に付け替えず、そのまま伝わる。"
  (let ((bad (nb::%make-traceable-function
              (list 'a)
              (lambda (a) (nb:tree-map (lambda (x y) (declare (ignore y)) x) (list a) (list a a))))))
    (signals nb:pytree-structure-mismatch
      (%ctl-traced-call (lambda (p a) (nb:cond* (funcall *ctl-lt* p 0.5d0) bad bad a))
                        (list (%ctl-scalar 0d0) (%ctl-scalar 1d0))))))

(test pytree-control/cond-with-a-constant-pred-calls-the-branch-with-the-trees
  "pred が T / NIL のとき、枝は木の operands をそのまま受け取る。"
  (let ((a (list (%ctl-scalar 1d0) (vector (%ctl-scalar 2d0)))))
    (is (%ctl-same-tree-p (nb:cond* t *ctl-then* *ctl-else* a a)
                          (funcall *ctl-then* a a)))
    (is (%ctl-same-tree-p (nb:cond* nil *ctl-then* *ctl-else* a a)
                          (funcall *ctl-else* a a)))))

;;; ---- scan ----

(defun %ctl-scan-body (carry x)
  "carry の葉ごとに tanh(c + s) (s は x の最初の葉の総和。x に葉が無ければ 0) を新しい carry とし、
ys は (x を 0.9 倍した木, 新しい carry)。"
  (let* ((xl (nb:tree-leaves x))
         (s (if xl (nb:reduce-sum (first xl)) 0d0))
         (new (nb:tree-map (lambda (c) (funcall *ctl-step* c s)) carry)))
    (values new (list (%ctl-scale-tree x) new))))

(defparameter *ctl-scan-f* (nb::%make-traceable-function (list 'carry 'x) #'%ctl-scan-body))

(defun %ctl-xs-tree (spec length)
  "SPEC の葉を、先頭の軸が LENGTH の f64 の配列に置き換えた木（左から 4 個より後ろの葉は NIL）。"
  (let ((count 0))
    (nb:tree-map (lambda (n)
                   (when (<= (incf count) 4)
                     (make-random-array
                      (make-array-spec (cons length (ecase (mod n 3) (0 '()) (1 '(3)) (2 '(2 3)))) :f64)
                      :seed (+ 2000 n))))
                 spec)))

(defun %ctl-row (array i)
  "ARRAY の先頭の軸の添字 I の行（新しい配列）。"
  (let* ((dims (rest (array-dimensions array)))
         (row (make-array dims :element-type (array-element-type array)))
         (size (array-total-size row)))
    (dotimes (j size row)
      (setf (row-major-aref row j) (row-major-aref array (+ (* i size) j))))))

(defun %ctl-stack (rows)
  "ROWS（同じ形の配列のリスト）を先頭の軸に積んだ配列。"
  (let* ((first-row (first rows))
         (size (array-total-size first-row))
         (out (make-array (cons (length rows) (array-dimensions first-row))
                          :element-type (array-element-type first-row))))
    (loop for row in rows for i from 0
          do (dotimes (j size) (setf (row-major-aref out (+ (* i size) j)) (row-major-aref row j))))
    out))

(defun %ctl-scan-reference (carry xs length reverse)
  "scan の仕様そのもの: 添字を辿って本体を呼び、y を添字の位置に積む。(values carry ys)。"
  (let ((ys (make-array length)))
    (dolist (i (if reverse (loop for i from (1- length) downto 0 collect i) (loop for i below length collect i)))
      (multiple-value-bind (new y)
          (%ctl-scan-body carry (nb:tree-map (lambda (x) (%ctl-row x i)) xs))
        (setf carry new (aref ys i) y)))
    (let ((treedef (nb:tree-structure (aref ys 0))))
      (values carry
              (nb:tree-unflatten
               treedef
               (apply #'mapcar (lambda (&rest column) (%ctl-stack column))
                      (map 'list #'nb:tree-leaves ys)))))))

(def-pytree-control-property pytree-control/scan-tree-carry-xs-ys-match-reference
  "木の init / xs / ys の scan は、eager でもトレースしても、Lisp のループによる参照実装と
同じ treedef・同じ値になる（reverse を含む）。ys の木は本体が返した形に積んだもの。"
  (carry-spec xs-spec length reverse)
  (generator (tuple (pytree-spec) (pytree-spec) (uniform-integer :lo 1 :hi 4) (uniform-integer :lo 0 :hi 1)))
  (let ((init (pytree-array-tree carry-spec :f64 :max-leaves 4))
        (xs (%ctl-xs-tree xs-spec length))
        (reverse (= reverse 1)))
    (or (not (%ctl-has-leaves-p init))
        (multiple-value-bind (expected-carry expected-ys) (%ctl-scan-reference init xs length reverse)
          (flet ((scan-values (init xs)
                   (multiple-value-bind (carry ys) (nb:scan *ctl-scan-f* init xs :length length :reverse reverse)
                     (values carry ys))))
            (multiple-value-bind (carry ys) (scan-values init xs)
              (multiple-value-bind (traced-carry traced-ys)
                  (%ctl-traced-call #'scan-values (list init xs))
                (and (%ctl-same-tree-p carry expected-carry)
                     (%ctl-same-tree-p ys expected-ys)
                     (%ctl-same-tree-p traced-carry expected-carry)
                     (%ctl-same-tree-p traced-ys expected-ys)))))))))

(test pytree-control/scan-rejects-carry-treedef-mismatch
  "本体が返す carry の木の形が init と違えば（葉の個数と aval が同じでも）SCAN-CARRY-MISMATCH。"
  (let ((init (list (%ctl-scalar 0d0) (%ctl-scalar 1d0))))
    (flet ((scan-with (body)
             (nb:scan (nb::%make-traceable-function (list 'c 'x) body) init '() :length 2)))
      (signals nb:scan-carry-mismatch (scan-with (lambda (c x) (declare (ignore x)) (values (list c) '()))))
      (signals nb:scan-carry-mismatch
        (scan-with (lambda (c x) (declare (ignore x)) (values (coerce c 'simple-vector) '()))))
      (signals nb:scan-carry-mismatch
        (scan-with (lambda (c x) (declare (ignore x)) (values (list (first c) (list (second c))) '())))))))

(test pytree-control/scan-accepts-a-bare-array-carry-and-xs
  "init と xs は配列そのもの（葉1つの木）でもよく、ys の木の形は本体が返した形になる。"
  (multiple-value-bind (carry ys)
      (nb:scan (nb::%make-traceable-function
                (list 'c 'x) (lambda (c x) (values (funcall *ctl-step* c (nb:reduce-sum x)) (vector c))))
               (%ctl-scalar 0d0) (%ctl-xs-tree 0 3) :length 3)
    (is (arrayp carry))
    (is (simple-vector-p ys))
    (is (equal '(3) (array-dimensions (svref ys 0))))))

;;; ---- grad / vmap ----

(defun %ctl-grad-function (xs length)
  "init の木を受け取り、scan の最終 carry と ys の葉の総和を返す TRACEABLE-FUNCTION。"
  (nb::%make-traceable-function
   (list 'init)
   (lambda (init)
     (multiple-value-bind (carry ys) (nb:scan *ctl-scan-f* init xs :length length)
       (funcall (pytree-loss-function) (list carry ys))))))

(def-pytree-control-property pytree-control/grad-through-tree-carry-scan-matches-central-difference
  "木の carry の scan を通る grad は、init と同じ treedef の木を返し、葉ごとに f64 の中心差分と一致する。"
  (carry-spec xs-spec length)
  (generator (tuple (pytree-spec) (pytree-spec) (uniform-integer :lo 1 :hi 3)))
  (let* ((init (pytree-array-tree carry-spec :f64 :max-leaves 3))
         (xs (%ctl-xs-tree xs-spec length))
         (f (%ctl-grad-function xs length))
         (treedef (nb:tree-structure init)))
    (or (not (%ctl-has-leaves-p init))
        (let ((grads (funcall (nb:grad f) init))
              (expected (central-difference-gradient
                         (lambda (&rest leaves) (funcall f (nb:tree-unflatten treedef leaves)))
                         (nb:tree-leaves init))))
          (and (nb:treedef= treedef (nb:tree-structure grads))
               (%results-close-p (nb:tree-leaves grads) expected
                                 :rtol *autodiff-rtol* :atol *autodiff-atol*))))))

(defun %ctl-vmap-function (length)
  "(h xs) → 木の carry (登録した節 + ベクタ + リスト) を持つ scan の結果の葉を多値で返す関数。"
  (nb::%make-traceable-function
   (list 'h 'xs)
   (lambda (h xs)
     (multiple-value-bind (carry ys)
         (nb:scan (nb::%make-traceable-function
                   (list 'c 'x)
                   (lambda (c x)
                     (let* ((s (nb:reduce-sum (svref (first x) 0)))
                            (left (funcall *ctl-step* (svref (pytree-test-pair-left c) 0) s))
                            (right (funcall *ctl-step* (first (pytree-test-pair-right c)) 0d0)))
                       (values (make-pytree-test-pair (vector left) (list right) 0)
                               (list left right)))))
                  (make-pytree-test-pair (vector h) (list (funcall *ctl-scale* h)) 0)
                  (list (vector xs)) :length length)
       (values-list (nb:tree-leaves (list carry ys)))))))

(def-pytree-control-property pytree-control/vmap-of-tree-carry-scan-matches-per-example-reference
  "木の carry の scan を vmap した結果は、バッチ軸で切り出した各要素に eager で適用して積み直した
ものと一致する（h と xs のバッチ軸は、両方・h だけ・xs だけ）。"
  (size seed length which)
  (generator (tuple (uniform-integer :lo 1 :hi 3) (uniform-integer :lo 0 :hi 100000)
                    (uniform-integer :lo 1 :hi 3) (uniform-integer :lo 0 :hi 2)))
  (let* ((axes (nth which '((0 0) (0 nil) (nil 0))))
         (h (make-random-array (make-array-spec (if (first axes) (list size 3) '(3)) :f64) :seed seed))
         (xs (make-random-array (make-array-spec (if (second axes) (list size length 2) (list length 2)) :f64)
                                :seed (1+ seed)))
         (f (%ctl-vmap-function length))
         (expected (reference-vmap f (list h xs) :in-axes axes))
         (actual (multiple-value-list (funcall (nb:vmap f :in-axes axes :out-axes 0) h xs))))
    (and (= (length actual) (length expected))
         (every (lambda (a e) (allclose a e :dtype :f64)) actual expected))))

(test pytree-control/vmap-accepts-a-tree-argument-for-a-tree-carry-scan
  "木（リストと登録した節）を vmap に直接渡した、木の carry の scan は、要素ごとに適用した結果と一致する。"
  (let* ((f (nb::%make-traceable-function
             (list 'tree 'xs)
             (lambda (tree xs)
               (multiple-value-bind (carry ys)
                   (nb:scan (nb::%make-traceable-function
                             (list 'c 'x)
                             (lambda (c x)
                               (let ((new (nb:tree-map (lambda (v) (funcall *ctl-step* v (nb:reduce-sum x))) c)))
                                 (values new new))))
                            tree xs)
                 (values-list (nb:tree-leaves (list carry ys)))))))
         (size 3)
         (tree (list (make-random-array (make-array-spec (list size 3) :f64) :seed 1)
                     (vector (make-random-array (make-array-spec (list size 2) :f64) :seed 2))))
         (xs (make-random-array (make-array-spec (list size 4 2) :f64) :seed 3))
         (actual (multiple-value-list (funcall (nb:vmap f :in-axes '(0 0)) tree xs)))
         (expected
           (let ((per-example
                   (loop for i below size
                         collect (multiple-value-list
                                  (funcall f (nb:tree-map (lambda (a) (%ctl-row a i)) tree)
                                           (%ctl-row xs i))))))
             (apply #'mapcar (lambda (&rest rows) (%ctl-stack rows)) per-example))))
    (is (= (length actual) (length expected)))
    (is (every (lambda (a e) (allclose a e :dtype :f64)) actual expected))))

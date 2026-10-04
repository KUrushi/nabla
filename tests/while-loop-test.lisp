;;;; while-loop プリミティブ（issue #131）。
;;;;
;;;; 性質: while-loop の結果は、Lisp のループで body-fn を回した結果と一致する
;;;; （0回で終わる場合を含む）。eager（配列）・トレース中・閉包で外側のトレーサを
;;;; 捕まえる場合のすべてで確かめる。整数 dtype は未導入なので、カウンタは f32 の
;;;; rank 0 で持つ（i を 1.0 ずつ増やして limit と比べる）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defparameter *wl-shapes* '(() (3) (2 3) (2 1 4)))

(defun %wl-scalar (x &optional (dtype :f32))
  (nb::%scalar-array x dtype))

(defun %wl-seed-array (seed shape dtype)
  (make-random-array (make-array-spec shape dtype) :seed seed))

(defparameter *wl-cond*
  (nb:with-tracing (c) (< (first c) (second c))))

(defparameter *wl-body*
  (nb:with-tracing (c)
    (list (+ (first c) 1.0) (second c) (+ (third c) (* (third c) 0.5)))))

(defun %wl-lisp-loop (cond-fn body-fn carries)
  "while-loop の仕様そのもの: cond-fn が真の間 body-fn を回す。"
  (loop while (= 1 (aref (funcall cond-fn carries)))
        do (setf carries (funcall body-fn carries)))
  carries)

(defun %wl-all-close (actual expected)
  (and (= (length actual) (length expected))
       (every (lambda (a e) (allclose a e :dtype :f32))
              actual expected)))

(test while-loop/eager-equals-lisp-loop
  "eager の while-loop は、Lisp のループで body-fn を回した結果と一致する。
limit は 0〜6（0回で終わる場合を含む）。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (let* ((limit (float (mod seed 7) 1.0))
                         (shape (nth (mod (floor seed 7) 4) *wl-shapes*))
                         (init (list (%wl-scalar 0.0) (%wl-scalar limit)
                                     (%wl-seed-array seed shape :f32))))
                    (%wl-all-close (nb:while-loop *wl-cond* *wl-body* init)
                                   (%wl-lisp-loop *wl-cond* *wl-body* init))))
                :regression-id while-loop/eager-equals-lisp-loop
                :regression-file (regression-path "while-loop-eager-equals-lisp-loop"))))

(test while-loop/zero-iterations-return-init
  "条件が最初から偽なら、init の値がそのまま返る。"
  (let* ((x (%wl-seed-array 3 '(2 3) :f32))
         (result (nb:while-loop *wl-cond* *wl-body*
                                (list (%wl-scalar 0.0) (%wl-scalar 0.0) x))))
    (is (= 3 (length result)))
    (is (equalp x (third result)))))

(test while-loop/carries-of-different-shapes-and-dtypes
  "形も dtype も違う carry（f32 の配列と f64 の配列）を同時に持てる。"
  (let ((cond-fn (nb:with-tracing (c) (< (first c) 4.0)))
        (body-fn (nb:with-tracing (c)
                   (list (+ (first c) 1.0) (+ (second c) (second c)) (* (third c) (third c))))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((init (list (%wl-scalar (float (mod seed 6) 1.0))
                                       (%wl-seed-array seed '(2 3) :f32)
                                       (%wl-seed-array (1+ seed) '(4) :f64)))
                           (actual (nb:while-loop cond-fn body-fn init))
                           (expected (%wl-lisp-loop cond-fn body-fn init)))
                      (and (allclose (first actual) (first expected) :dtype :f32)
                           (allclose (second actual) (second expected) :dtype :f32)
                           (allclose (third actual) (third expected) :dtype :f64))))
                  )
          "形・dtype の違う carry の結果が Lisp のループと一致しなかった")))

(defun %wl-traced-graph ()
  "limit・step・x を引数に、limit を閉包で（cond から）、step を閉包で（body から）
捕まえる while-loop をトレースした graph。"
  (nb::trace-to-graph
   (nb:with-tracing (limit step x)
     (let ((result (nb:while-loop
                    (nb:with-tracing (c) (< (first c) limit))
                    (nb:with-tracing (c) (list (+ (first c) 1.0) (+ (second c) step)))
                    (list (%wl-scalar 0.0) x))))
       (values (first result) (second result))))
   (list (nb:make-aval '() :f32) (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32))))

(defun %wl-while-eqns (graph)
  (remove-if-not (lambda (e) (eq :while-loop (nb::primitive-name (nb:eqn-prim e))))
                 (nb:graph-eqns graph)))

(test while-loop/traced-graph-has-one-while-eqn-with-captured-operands
  "トレース中の while-loop は :while-loop の eqn を1つ足し、閉包で捕まえた外側の
値（limit と step）は loop 不変の追加のオペランドとして eqn の入力の末尾に並ぶ。"
  (let* ((graph (%wl-traced-graph))
         (eqns (%wl-while-eqns graph)))
    (is (= 1 (length eqns)))
    ;; carry 2 + captured 2（limit は cond、step は body）
    (is (= 4 (length (nb:eqn-invars (first eqns)))))
    (is (= 4 (length (nb:eqn-outvars (first eqns)))))))

(test while-loop/traced-equals-lisp-loop-with-closure-capture
  "閉包で外側のトレーサを捕まえた while-loop をトレースして eval-graph した結果は、
Lisp のループの結果と一致する（limit が 0 の場合を含む）。"
  (let ((graph (%wl-traced-graph)))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((n (mod seed 7))
                           (limit (%wl-scalar (float n 1.0)))
                           (step (%wl-seed-array seed '(3) :f32))
                           (x (%wl-seed-array (1+ seed) '(3) :f32))
                           (actual (multiple-value-list (nb:eval-graph graph limit step x)))
                           (expected-x (let ((v x))
                                         (dotimes (_ n v) (setf v (nb::%t-add v step))))))
                      (and (allclose (first actual) (%wl-scalar (float n 1.0)) :dtype :f32)
                           (allclose (second actual) expected-x :dtype :f32))))
                  :regression-id while-loop/traced-closure
                  :regression-file (regression-path "while-loop-traced-closure")))))

;;; ---- cond と body が同じ外側の値を捕まえる（issue #166 の d）。捕捉値は重複なく
;;; 1回だけオペランドになる ----

(defun %wl-shared-capture-graph ()
  "limit を cond と body の両方が、step を body だけが閉包で捕まえる while-loop の graph。
body は limit を2回使う。"
  (nb::trace-to-graph
   (nb:with-tracing (limit step x)
     (let ((result (nb:while-loop
                    (nb:with-tracing (c) (< (first c) limit))
                    (nb:with-tracing (c)
                      (list (+ (first c) 1.0) (+ (second c) (* step limit) limit)))
                    (list (%wl-scalar 0.0) x))))
       (values (first result) (second result))))
   (list (nb:make-aval '() :f32) (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32))))

(test while-loop/shared-capture-is-one-operand
  "cond と body が同じ外側のトレーサを捕まえても、eqn のオペランド（と StableHLO の
while の carry）にはその値が1回だけ現れる: carry 2 + 捕捉値 2（limit と step）。"
  (let* ((graph (%wl-shared-capture-graph))
         (eqn (first (%wl-while-eqns graph)))
         (operands (nb:eqn-invars eqn)))
    (is (= 4 (length operands)))
    (is (= 4 (length (remove-duplicates operands :test #'eq))))
    (is (= 4 (length (nb:eqn-outvars eqn))))
    (is (= 4 (length (nb:graph-invars (getf (nb:eqn-params eqn) :cond)))))
    (is (= 4 (length (nb:graph-invars (getf (nb:eqn-params eqn) :body)))))
    (is (search "-> (tensor<f32>, tensor<3xf32>, tensor<f32>, tensor<3xf32>)"
                (nb:emit-stablehlo graph)))))

(test while-loop/shared-capture-equals-lisp-loop
  "cond と body が同じ外側の値を捕まえる while-loop をトレースして eval-graph した
結果は、Lisp のループの結果（x に step*limit + limit を limit 回足す）と一致する。"
  (let ((graph (%wl-shared-capture-graph)))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((n (mod seed 7))
                           (limit (%wl-scalar (float n 1.0)))
                           (step (%wl-seed-array seed '(3) :f32))
                           (x (%wl-seed-array (1+ seed) '(3) :f32))
                           (actual (multiple-value-list (nb:eval-graph graph limit step x)))
                           (delta (funcall (nb:with-tracing (s l) (+ (* s l) l)) step limit))
                           (expected-x (let ((v x))
                                         (dotimes (_ n v) (setf v (nb::%t-add v delta))))))
                      (and (allclose (first actual) limit :dtype :f32)
                           (allclose (second actual) expected-x :dtype :f32))))
                  :regression-id while-loop/shared-capture
                  :regression-file (regression-path "while-loop-shared-capture")))))

(test while-loop/array-init-inside-a-trace
  "外側のトレースの中で、init が配列（定数）のときも while-loop をトレースできる。"
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (x)
                  (second (nb:while-loop *wl-cond* *wl-body*
                                         (list (%wl-scalar 0.0) (%wl-scalar 2.0) x))))
                (list (nb:make-aval '(2) :f32)))))
    (is (= 1 (length (%wl-while-eqns graph))))
    (let ((x (%wl-seed-array 5 '(2) :f32)))
      (is (allclose (nb:eval-graph graph x)
                    (second (%wl-lisp-loop *wl-cond* *wl-body*
                                           (list (%wl-scalar 0.0) (%wl-scalar 2.0) x)))
                    :dtype :f32)))))

(test while-loop/emits-stablehlo-while-with-regions
  "StableHLO は stablehlo.while の2つのリージョン（cond と body）で、ブロック引数を持つ。"
  (let ((text (nb:emit-stablehlo (%wl-traced-graph))))
    (is (search "\"stablehlo.while\"" text))
    (is (= 2 (count-if (lambda (l) (search "^bb0(" l)) (%sg-lines text))))
    (is (= 2 (count-if (lambda (l) (search "stablehlo.return" l)) (%sg-lines text))))))

;;; ---- 厳格な検査（文書化したコンディション） ----

(test while-loop/rejects-non-list-init
  "init がリストでない・空・要素が配列でない、または関数でないものを渡すと
WHILE-LOOP-ARGUMENT-ERROR。"
  (signals nb:while-loop-argument-error
    (nb:while-loop *wl-cond* *wl-body* (%wl-scalar 0.0)))
  (signals nb:while-loop-argument-error
    (nb:while-loop *wl-cond* *wl-body* (vector (%wl-scalar 0.0))))
  (signals nb:while-loop-argument-error
    (nb:while-loop *wl-cond* *wl-body* '()))
  (signals nb:while-loop-argument-error
    (nb:while-loop *wl-cond* *wl-body* (list 1.0 2.0 3.0)))
  (signals nb:while-loop-argument-error
    (nb:while-loop *wl-cond* 3 (list (%wl-scalar 0.0))))
  ;; dtype を決められない配列（生の (unsigned-byte 16) = bf16 / f16）も同じコンディション
  (signals nb:while-loop-argument-error
    (nb:while-loop *wl-cond* *wl-body*
                   (list (make-array '() :element-type '(unsigned-byte 16)) (%wl-scalar 1.0)
                         (%wl-scalar 1.0)))))

(test while-loop/signals-carry-mismatch-at-trace-time
  "body-fn の出力の aval（個数・shape・dtype）が init と違えば WHILE-LOOP-CARRY-MISMATCH。
本体が1回も実行されない場合（cond が最初から偽）でも、トレース時に検出する。"
  (let ((cond-fn (nb:with-tracing (c) (< (first c) 0.0)))
        (init (list (%wl-scalar 0.0) (%wl-seed-array 1 '(3) :f32))))
    (signals nb:while-loop-carry-mismatch
      (nb:while-loop cond-fn (nb:with-tracing (c) (list (first c))) init))
    (signals nb:while-loop-carry-mismatch
      (nb:while-loop cond-fn (nb:with-tracing (c) (list (first c) (nb:reduce-sum (second c)))) init))
    (signals nb:while-loop-carry-mismatch
      (nb:while-loop cond-fn (nb:with-tracing (c) (list (first c) (nb:convert (second c) :f64))) init))
    (signals nb:while-loop-carry-mismatch
      (nb:while-loop cond-fn (nb:with-tracing (c) (list (first c) (second c) (second c))) init))
    (signals nb:while-loop-argument-error
      (nb:while-loop cond-fn (nb:with-tracing (c) (first c)) init))))

(test while-loop/signals-condition-error-when-cond-is-not-scalar-i1
  "cond-fn の結果が rank 0 の :i1 でなければ WHILE-LOOP-CONDITION-ERROR。"
  (let ((init (list (%wl-scalar 0.0) (%wl-seed-array 1 '(3) :f32)))
        (body (nb:with-tracing (c) (list (+ (first c) 1.0) (second c)))))
    (signals nb:while-loop-condition-error
      (nb:while-loop (nb:with-tracing (c) (first c)) body init))
    (signals nb:while-loop-condition-error
      (nb:while-loop (nb:with-tracing (c) (< (second c) 1.0)) body init))))

(test while-loop/all-conditions-are-while-loop-errors
  "引数・carry・cond の3つのコンディションは WHILE-LOOP-ERROR の子で、ERROR である。"
  (dolist (name '(nb:while-loop-argument-error nb:while-loop-carry-mismatch nb:while-loop-condition-error))
    (is (subtypep name 'nb:while-loop-error)))
  (is (subtypep 'nb:while-loop-error 'error)))

(test while-loop/grad-signals-autodiff-error-naming-the-primitive
  "grad が while-loop を通ると（逆モードは対応しないので）AUTODIFF-ERROR が出て、原因の
プリミティブ名 :WHILE-LOOP をメッセージで報告する（前進モードの jvp は #134 で対応済み）。"
  (let ((f (nb:with-tracing (x)
             (nb:reduce-sum (second (nb:while-loop *wl-cond* *wl-body*
                                                (list (%wl-scalar 0.0) (%wl-scalar 2.0) x)))))))
    (handler-case (funcall (nb:grad f) (%wl-seed-array 1 '(3) :f32))
      (nb:autodiff-error (c)
        (is (search ":WHILE-LOOP" (princ-to-string c))))
      (:no-error (&rest values)
        (declare (ignore values))
        (fail "grad が while-loop を通ったのにエラーにならなかった")))))

(test while-loop/rejects-non-function-cond-fn
  "COND-FN が関数でなければ WHILE-LOOP-ARGUMENT-ERROR。"
  (signals nb:while-loop-argument-error
    (nb:while-loop 3 *wl-body* (list (%wl-scalar 0.0) (%wl-scalar 1.0) (%wl-scalar 1.0)))))

(test while-loop/rejects-tracer-of-a-finished-trace
  "終わったトレースのトレーサを init に混ぜると、トレースの外から呼んでも TRACING-ERROR。"
  (let* ((stale nil)
         (remember (lambda (x) (setf stale x))))
    (nb::trace-to-graph (nb:with-tracing (x) (funcall remember x) x) (list (nb:make-aval '() :f32)))
    (signals nb::tracing-error
      (nb:while-loop *wl-cond* *wl-body* (list stale (%wl-scalar 1.0) (%wl-scalar 1.0))))))

(defun %wl-eqn-with (&key (n-carries 2) body-out-avals cond-in-avals cond-out-aval not-pass-through pass-all)
  "abstract-eval の検査を直接確かめるための :while-loop の eqn（既定は妥当な組）を作る。
carry 2 個（f32 の rank 0）。"
  (let* ((a (nb:make-aval '() :f32))
         (i1 (nb:make-aval '() :i1))
         (avals (list a a))
         (cond-vars (mapcar #'nb::make-var (or cond-in-avals avals)))
         (cond-out (nb::make-var (or cond-out-aval i1)))
         (body-vars (mapcar #'nb::make-var avals))
         (body-outs (cond (pass-all body-vars) (not-pass-through
                        ;; 2番目の出力（n-carries = 1 のとき素通しのはず）を別の var にする
                        (list (first body-vars) (nb::make-var a)))
                        (t (mapcar #'nb::make-var (or body-out-avals avals)))))
         (vars (mapcar #'nb::make-var avals)))
    (apply #'nb::make-eqn :while-loop vars
           (list :cond (nb::make-graph cond-vars '() (list cond-out))
                 :body (nb::make-graph body-vars '() body-outs)
                 :n-carries n-carries))))

(test while-loop/abstract-eval-validates-its-params
  "abstract-eval は、cond / body の入出力の aval と n-carries が eqn の入力と整合しない
params を PRIMITIVE-ERROR にする（GRAPH を手で組んだときの防御）。"
  (let ((a (nb:make-aval '() :f32)) (b (nb:make-aval '(2) :f32)))
    (is (not (null (%wl-eqn-with))))
    (is (not (null (%wl-eqn-with :n-carries 0 :pass-all t))))
    (is (not (null (%wl-eqn-with :n-carries 2))))
    (signals nb:primitive-error (%wl-eqn-with :n-carries -1))
    (signals nb:primitive-error (%wl-eqn-with :n-carries 3))
    (signals nb:primitive-error (%wl-eqn-with :cond-in-avals (list a b)))
    (signals nb:primitive-error (%wl-eqn-with :cond-out-aval a))
    (signals nb:primitive-error (%wl-eqn-with :body-out-avals (list a b)))
    ;; carry 以降の出力は、同じ入力 var の素通しでなければならない
    (signals nb:primitive-error (%wl-eqn-with :n-carries 1 :not-pass-through t))))

(test while-loop/abstract-eval-requires-pass-through-of-invariant-operands
  "carry 以降の出力が入力の素通しでない body は PRIMITIVE-ERROR。素通し（同じ var）なら通る。"
  (let ((a (nb:make-aval '() :f32)))
    (is (not (null (let* ((x (nb::make-var a)) (y (nb::make-var a))
                          (c (nb::make-graph (list x y) '() (list (nb::make-var (nb:make-aval '() :i1)))))
                          (b (nb::make-graph (list x y) '() (list x y))))
                     (nb::make-eqn :while-loop (list (nb::make-var a) (nb::make-var a))
                                   :cond c :body b :n-carries 1)))))
    (signals nb:primitive-error (%wl-eqn-with :n-carries 1 :not-pass-through t))))

;;; ---- 整数・:i1 の carry、入れ子、捕捉値の除去 ----

(defun %wl-i32 (n) (nb::%scalar-array n :i32))

(test while-loop/i32-counter-equals-lisp-loop
  "整数（:i32）のカウンタでも、結果は Lisp のループと一致する（0回を含む）。"
  (let ((cond-fn (nb:with-tracing (c) (< (first c) (second c))))
        (body-fn (nb:with-tracing (c) (list (+ (first c) 1) (second c) (+ (third c) (third c))))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((init (list (%wl-i32 0) (%wl-i32 (mod seed 7))
                                       (%wl-seed-array seed '(3) :f32)))
                           (actual (nb:while-loop cond-fn body-fn init))
                           (expected (%wl-lisp-loop cond-fn body-fn init)))
                      (and (equalp (first actual) (first expected))
                           (allclose (third actual) (third expected) :dtype :f32)))))
          "i32 カウンタの結果が Lisp のループと一致しなかった")))

(test while-loop/i1-carry-equals-lisp-loop
  "carry に :i1 の値（フラグ）を持てる。結果が Lisp のループと一致する。"
  (let ((cond-fn (nb:with-tracing (c) (second c)))
        (body-fn (nb:with-tracing (c) (list (+ (first c) 1.0) (< (+ (first c) 1.0) 5.0)))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((n (%wl-scalar (float (mod seed 8) 1.0)))
                           (init (list n (%wl-flag n)))
                           (actual (nb:while-loop cond-fn body-fn init))
                           (expected (%wl-lisp-loop cond-fn body-fn init)))
                      (and (allclose (first actual) (first expected) :dtype :f32)
                           (equalp (second actual) (second expected)))))))))

(test while-loop/nested-while-captures-from-grandparent
  "入れ子の while の内側が、祖父母のトレースの値を捕まえても、外側の結果と一致する。"
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (limit x)
                  (first
                   (nb:while-loop
                    (nb:with-tracing (c) (< (second c) 3.0))
                    (nb:with-tracing (c)
                      (list (first (nb:while-loop
                                    (nb:with-tracing (d) (< (second d) 2.0))
                                    (nb:with-tracing (d) (list (+ (first d) limit) (+ (second d) 1.0)))
                                    (list (first c) (%wl-scalar 0.0))))
                            (+ (second c) 1.0)))
                    (list x (%wl-scalar 0.0)))))
                (list (nb:make-aval '() :f32) (nb:make-aval '() :f32)))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let ((limit (%wl-seed-array seed '() :f32))
                          (x (%wl-seed-array (1+ seed) '() :f32)))
                      ;; 外側 3 回 × 内側 2 回 = limit を6回足す
                      (allclose (nb:eval-graph graph limit x)
                                (let ((v x)) (dotimes (_ 6 v) (setf v (nb::%t-add v limit))))
                                :dtype :f32)))))))

(test while-loop/captured-outputs-are-stripped-from-the-result
  "捕まえた値の素通しの出力は、公開の結果には含まれない（carry の個数だけ返る）。"
  (nb::trace-to-graph
   (nb:with-tracing (limit step x)
     (let ((result (nb:while-loop (nb:with-tracing (c) (< (first c) limit))
                                  (nb:with-tracing (c) (list (+ (first c) 1.0) (+ (second c) step)))
                                  (list (%wl-scalar 0.0) x))))
       (is (= 2 (length result)))
       (first result)))
   (list (nb:make-aval '() :f32) (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32))))

(defun %wl-flag (n)
  "rank 0 の f32 配列 N から、:i1 の rank 0 配列 (n < 5) を作る（eager）。"
  (funcall (nb:with-tracing (v) (< v 5.0)) n))

;;; ---- 定数のオペランドは optimization_barrier を通す（IREE 3.11 のコンパイラのバグの回避。
;;; docs/stablehlo-ops.md の制御構造の節） ----

(defun %wl-count-subseq (needle haystack)
  (loop with start = 0
        for pos = (search needle haystack :start2 start)
        while pos count t do (setf start (1+ pos))))

(defun %wl-const-operand-graph ()
  "carry 3 つのうち、カウンタと y の初期値が定数、x だけが引数の while-loop。"
  (nb::trace-to-graph
   (nb:with-tracing (limit x)
     (let ((result (nb:while-loop
                    (nb:with-tracing (c) (< (first c) limit))
                    (nb:with-tracing (c)
                      (list (+ (first c) 1.0) (* (second c) 0.5) (+ (third c) 1.0)))
                    (list (%wl-scalar 0.0) x (%wl-scalar 0.0)))))
       (values (third result) (second result))))
   (list (nb:make-aval '() :f32) (nb:make-aval '(3) :f32))))

(test while-loop/emit-routes-constant-operands-through-optimization-barrier
  "while のオペランドのうち graph の定数（stablehlo.constant）のものだけが、while の前の
stablehlo.optimization_barrier を通る（引数の x と limit は通らない）。値は変わらない
（eval-graph の結果は Lisp のループと一致する）。"
  (let* ((graph (%wl-const-operand-graph))
         (text (nb:emit-stablehlo graph))
         (lines (%sg-lines text))
         (barrier (find-if (lambda (l) (search "stablehlo.optimization_barrier" l)) lines)))
    (is (not (null barrier)))
    ;; 定数 2 つ（カウンタと y）の分だけ、barrier は 2 つのオペランドと 2 つの結果を持つ
    (is (= 2 (count #\% (subseq barrier (1+ (position #\= barrier)) (position #\: barrier)))))
    (is (= 1 (count-if (lambda (l) (search "stablehlo.optimization_barrier" l)) lines)))
    (let ((while-line (find-if (lambda (l) (search "\"stablehlo.while\"(" l)) lines)))
      ;; while のオペランドは、barrier の結果（%wbar...）と引数（%0 = limit、%1 = x）
      (is (= 2 (%wl-count-subseq "%wbar" while-line)))
      (is (search "%1" while-line)))
    (let ((x (%wl-seed-array 3 '(3) :f32)))
      (is (allclose (second (multiple-value-list (nb:eval-graph graph (%wl-scalar 2.0) x)))
                    (nb::%t-mul (%wl-seed-array 3 '(3) :f32) (nb::%scalar-array 0.25 :f32))
                    :dtype :f32)))))

(test while-loop/emit-has-no-barrier-without-constant-operands
  "オペランドがすべて引数なら、optimization_barrier は出ない。"
  (let ((text (nb:emit-stablehlo
               (nb::trace-to-graph
                (nb:with-tracing (i x)
                  (first (nb:while-loop (nb:with-tracing (c) (< (first c) 3.0))
                                        (nb:with-tracing (c) (list (+ (first c) 1.0) (second c)))
                                        (list i x))))
                (list (nb:make-aval '() :f32) (nb:make-aval '(3) :f32))))))
    (is (null (search "optimization_barrier" text)))))

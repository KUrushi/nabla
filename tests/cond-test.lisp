;;;; cond* プリミティブ（issue #130）の small テスト。
;;;;
;;;; 性質: cond* の結果は、選ばれた枝を直接呼んだ結果と一致する。eager では
;;;; 選ばれなかった枝は評価されない（副作用で確かめる）。aval の不一致・pred が
;;;; rank 0 の :i1 でないときは cond-error。IREE での実行は tests/iree/cond-test.lisp。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defvar *cond-test-boom-count* 0
  "%COND-TEST-BOOM の :eager が呼ばれた回数。")

;;; 選ばれなかった枝が eager で評価されないことを見るための、評価されたら
;;; エラーになる恒等プリミティブ（トレース時は abstract-eval だけで eager は呼ばれない）。
(nb:defprimitive %cond-test-boom ()
  :abstract-eval (lambda (in-avals) (first in-avals))
  :eager (lambda (arrays in-avals)
           (declare (ignore arrays in-avals))
           (incf *cond-test-boom-count*)
           (error "選ばれなかった枝が評価された")))

(defun %cond-pred (bit)
  (make-array '() :element-type 'bit :initial-element bit))

(defun %cond-avals (shape)
  (list (nb:make-aval '() :i1) (nb:make-aval shape :f32) (nb:make-aval shape :f32)))

(defun %cond-arrays (seed shape)
  (list (%cond-pred (mod seed 2))
        (make-random-array (make-array-spec shape :f32) :seed (+ seed 1))
        (make-random-array (make-array-spec shape :f32) :seed (+ seed 2))))

(defparameter *cond-shape-generator*
  (generator (tuple (uniform-integer :lo 1 :hi 4) (uniform-integer :lo 1 :hi 4))))

(defun %cond-graph (shape)
  "p ? (x + y, x * y) : (x - y, x) （複数出力）の graph。"
  (nb::trace-to-graph
   (nb:with-tracing (p x y)
     (multiple-value-bind (a b)
         (nb:cond* p
                   (nb:with-tracing (u v) (values (+ u v) (* u v)))
                   (nb:with-tracing (u v) (values (- u v) u))
                   x y)
       (values a b)))
   (%cond-avals shape)))

(test cond/result-equals-selected-branch-called-directly
  "cond* の結果は、選ばれた枝を直接呼んだ結果と一致する（PBT。形状・pred・複数出力）。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 100000) *cond-shape-generator*))
                (lambda (input)
                  (destructuring-bind (seed shape) input
                    (let* ((graph (%cond-graph shape))
                           (arrays (%cond-arrays seed shape))
                           (x (second arrays))
                           (y (third arrays))
                           (expected (if (zerop (mod seed 2))
                                         (list (reference-sub x y) x)
                                         (list (reference-add x y) (reference-mul x y))))
                           (actual (multiple-value-list (apply #'nb:eval-graph graph arrays))))
                      (and (= 2 (length actual))
                           (every (lambda (a e) (allclose a e :dtype :f32)) actual expected)))))
                :regression-id cond/equals-selected-branch
                :regression-file (regression-path "cond-equals-selected-branch"))))

(test cond/captured-outer-tracers-become-operands
  "枝が閉包で捕まえた外側のトレーサは、cond の eqn の invars の末尾に足され、結果も正しい。"
  (let* ((graph (nb::trace-to-graph
                 (nb:with-tracing (p x y)
                   (nb:cond* p
                             (nb:with-tracing (u) (+ u y))
                             (nb:with-tracing (u) (* u x))
                             x))
                 (%cond-avals '(2 3))))
         (eqn (first (nb:graph-eqns graph))))
    (is (eq :cond (nb::primitive-name (nb:eqn-prim eqn))))
    ;; pred, x（operand）, y（then の捕捉）, x（else の捕捉）
    (is (= 4 (length (nb:eqn-invars eqn))))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (destructuring-bind (p x y) (%cond-arrays seed '(2 3))
                      (allclose (nb:eval-graph graph p x y)
                                (if (zerop (aref p)) (reference-mul x x) (reference-add x y))
                                :dtype :f32)))))))

(test cond/eager-evaluates-only-the-selected-branch
  "eager（eval-graph）は選ばれた枝だけを評価する。選ばれなかった枝のエラーは起きない。"
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (p x)
                  (nb:cond* p
                            (nb:with-tracing (u) (+ u 1))
                            (nb:with-tracing (u) (nb::%trace-eqn :%cond-test-boom (list u)))
                            x))
                (list (nb:make-aval '() :i1) (nb:make-aval '(3) :f32))))
        (x (make-random-array (make-array-spec '(3) :f32) :seed 7))
        (*cond-test-boom-count* 0))
    (is (allclose (nb:eval-graph graph (%cond-pred 1) x) (reference-add x (make-array 3 :element-type 'single-float :initial-element 1f0))
                  :dtype :f32))
    (is (= 0 *cond-test-boom-count*))
    (signals error (nb:eval-graph graph (%cond-pred 0) x))
    (is (= 1 *cond-test-boom-count*))))

(test cond/non-tracer-pred-calls-the-branch-directly
  "pred が T / NIL / rank 0 の bit 配列（トレーサでない）なら、選ばれた枝をそのまま
呼び、cond の eqn は作らない。"
  (let ((x (make-random-array (make-array-spec '(2) :f32) :seed 3))
        (then (nb:with-tracing (u) (+ u u)))
        (else (nb:with-tracing (u) (* u u))))
    (is (allclose (nb:cond* t then else x) (reference-add x x) :dtype :f32))
    (is (allclose (nb:cond* nil then else x) (reference-mul x x) :dtype :f32))
    (is (allclose (nb:cond* (%cond-pred 1) then else x) (reference-add x x) :dtype :f32))
    (let ((graph (nb::trace-to-graph
                  (nb:with-tracing (x) (nb:cond* t then else x))
                  (list (nb:make-aval '(2) :f32)))))
      (is (equal '(:add) (mapcar (lambda (e) (nb::primitive-name (nb:eqn-prim e)))
                                 (nb:graph-eqns graph)))))))

;;; ---- eager の cond* は入力を書き換えない（issue #166） ----

(defparameter *cond-host-constant*
  (make-array 2 :element-type 'single-float :initial-contents '(7.0 8.0))
  "*COND-ALIAS-BRANCHES* の枝が閉包で捕まえるホストの配列。")

(defparameter *cond-alias-branches*
  (list
   (nb:with-tracing (u v) (values u v))                 ; operands をそのまま返す
   (nb:with-tracing (u v) (values *cond-host-constant* v)) ; 捕まえたホストの配列を返す
   (nb:with-tracing (u v) (values (+ u v) (* u v))))    ; 算術
  "どれも [2] の f32 を2つ受け取り2つ返す枝。入力をそのまま出力に回すものを含む。")

(defun %cond-snapshot (arrays)
  "ARRAYS の各配列の中身を写した新しい配列のリスト。"
  (mapcar (lambda (a)
            (let ((copy (make-array (array-dimensions a) :element-type (array-element-type a))))
              (dotimes (j (array-total-size a) copy)
                (setf (row-major-aref copy j) (row-major-aref a j)))))
          arrays))

(test cond/eager-does-not-modify-inputs
  "eager の cond* は operands・枝が捕まえたホストの配列を書き換えず、結果は選ばれた枝を
直接呼んだ結果と一致する。pred が T / NIL / bit 配列の直接の経路と、トレースした :cond を
eval-graph する経路の両方。結果が入力と EQ かどうかは問わない（README「配列の不変性」）。"
  (is (check-it
       (generator (tuple (integer 0 100000) (integer 0 2) (integer 0 2) (integer 0 2)))
       (lambda (case)
         (destructuring-bind (seed then-index else-index path) case
           (let* ((then (nth then-index *cond-alias-branches*))
                  (else (nth else-index *cond-alias-branches*))
                  (bit (mod seed 2))
                  (x (make-random-array (make-array-spec '(2) :f32) :seed seed))
                  (y (make-random-array (make-array-spec '(2) :f32) :seed (1+ seed)))
                  (inputs (list x y *cond-host-constant*))
                  (before (%cond-snapshot inputs))
                  (actual
                    (multiple-value-list
                     (ecase path
                       (0 (nb:cond* (= bit 1) then else x y))
                       (1 (nb:cond* (%cond-pred bit) then else x y))
                       (2 (nb:eval-graph
                               (nb::trace-to-graph
                                (nb:with-tracing (p a b) (nb:cond* p then else a b))
                                (list (nb:make-aval '() :i1) (nb:make-aval '(2) :f32)
                                      (nb:make-aval '(2) :f32)))
                               (%cond-pred bit) x y)))))
                  (expected (multiple-value-list
                             (apply (if (= bit 1) then else) (%cond-snapshot (list x y))))))
             (and (every #'equalp before inputs)
                  (= 2 (length actual))
                  (every (lambda (a e) (allclose a e :dtype :f32)) actual expected)))))
       :regression-id cond/eager-does-not-modify-inputs
       :regression-file (regression-path "cond-eager-does-not-modify-inputs"))))

(test cond/error-on-branch-aval-mismatch
  "両枝の出力の aval（個数・形状・dtype）が違えば、トレース時に cond-error。"
  (let ((avals (list (nb:make-aval '() :i1) (nb:make-aval '(2 3) :f32))))
    (signals nb:cond-error
      (nb::trace-to-graph
       (nb:with-tracing (p x)
         (nb:cond* p (nb:with-tracing (u) u) (nb:with-tracing (u) (nb:reduce-sum u :axes '(0))) x))
       avals))
    (signals nb:cond-error
      (nb::trace-to-graph
       (nb:with-tracing (p x)
         (nb:cond* p (nb:with-tracing (u) (values u u)) (nb:with-tracing (u) u) x))
       avals))
    (signals nb:cond-error
      (nb::trace-to-graph
       (nb:with-tracing (p x)
         (nb:cond* p (nb:with-tracing (u) u) (nb:with-tracing (u) (nb:convert u :f64)) x))
       avals))))

(test cond/error-on-bad-pred
  "pred が rank 0 の :i1 でなければ cond-error（トレーサの場合）。"
  (signals nb:cond-error
    (nb::trace-to-graph
     (nb:with-tracing (p x) (nb:cond* p (nb:with-tracing (u) u) (nb:with-tracing (u) u) x))
     (list (nb:make-aval '(2) :i1) (nb:make-aval '(2) :f32))))
  (signals nb:cond-error
    (nb::trace-to-graph
     (nb:with-tracing (p x) (nb:cond* p (nb:with-tracing (u) u) (nb:with-tracing (u) u) x))
     (list (nb:make-aval '() :f32) (nb:make-aval '(2) :f32)))))

(test cond/error-on-non-traceable-branch
  "枝が TRACEABLE-FUNCTION でなければ cond-error。"
  (signals nb:cond-error
    (nb::trace-to-graph
     (nb:with-tracing (p x) (nb:cond* p (lambda (u) u) (nb:with-tracing (u) u) x))
     (list (nb:make-aval '() :i1) (nb:make-aval '(2) :f32)))))

(test cond/stablehlo-is-if-with-two-regions
  "StableHLO は stablehlo.if の2つのリージョンになる。"
  (let ((text (nb:emit-stablehlo (%cond-graph '(2 3)))))
    (is (search "\"stablehlo.if\"" text))
    (is (= 2 (count-if (lambda (l) (search "stablehlo.return" l))
                       (%sg-lines text))))))

(test cond/array-and-number-operands-are-lifted-under-a-tracer-pred
  "pred がトレーサのとき、operand に配列・実数を渡すと定数としてリフトされる。"
  (let* ((arr (make-random-array (make-array-spec '(2) :f32) :seed 5))
         (graph (nb::trace-to-graph
                 (nb:with-tracing (p x)
                   (nb:cond* p
                             (nb:with-tracing (u a n) (+ (+ u a) n))
                             (nb:with-tracing (u a n) (- (- u a) n))
                             x arr 2.0))
                 (list (nb:make-aval '() :i1) (nb:make-aval '(2) :f32))))
         (x (make-random-array (make-array-spec '(2) :f32) :seed 6))
         (two (make-array 2 :element-type 'single-float :initial-element 2f0)))
    (is (allclose (nb:eval-graph graph (%cond-pred 1) x)
                  (reference-add (reference-add x arr) two) :dtype :f32))
    (is (allclose (nb:eval-graph graph (%cond-pred 0) x)
                  (reference-sub (reference-sub x arr) two) :dtype :f32))
    (signals nb:cond-error
      (nb::trace-to-graph
       (nb:with-tracing (p x) (nb:cond* p (nb:with-tracing (u v) u) (nb:with-tracing (u v) u) x "bad"))
       (list (nb:make-aval '() :i1) (nb:make-aval '(2) :f32))))))

(test cond/bad-non-tracer-pred-is-rejected
  "トレーサでない pred は T / NIL / rank 0 の bit 配列だけ。rank 1 の bit 配列、:f32 の配列、
その他の値は cond-error。bit 配列の 0 は else を選ぶ。"
  (let ((then (nb:with-tracing (u) (+ u u)))
        (else (nb:with-tracing (u) (* u u)))
        (x (make-random-array (make-array-spec '(2) :f32) :seed 3)))
    (is (allclose (nb:cond* (%cond-pred 0) then else x) (reference-mul x x) :dtype :f32))
    (signals nb:cond-error (nb:cond* (make-array 2 :element-type 'bit) then else x))
    (signals nb:cond-error (nb:cond* (make-array '() :element-type 'single-float) then else x))
    (signals nb:cond-error (nb:cond* 1 then else x))))

(defun %cond-eqn-avals-error-p (in-avals &rest params)
  (handler-case (progn (apply #'nb::make-eqn :cond (mapcar #'nb::make-var in-avals) params) nil)
    (nb::primitive-error () t)))

(test cond/primitive-abstract-eval-rejects-inconsistent-eqns
  "make-eqn で直接 :cond の eqn を作ったときも、不整合は primitive-error になる
（pred の型・枝の入力 aval の不一致・出力 aval の不一致・graph でない枝）。"
  (let* ((f32 (nb:make-aval '(2) :f32))
         (one (nb::trace-to-graph (nb:with-tracing (u) (+ u u)) (list f32)))
         (two (nb::trace-to-graph (nb:with-tracing (u) (values u u)) (list f32))))
    (let ((i1 (nb:make-aval '() :i1)))
      (is (not (%cond-eqn-avals-error-p (list i1 f32) :then one :else one)))
      (is (%cond-eqn-avals-error-p (list f32 f32) :then one :else one))
      (is (%cond-eqn-avals-error-p (list (nb:make-aval '(1) :i1) f32) :then one :else one))
      (is (%cond-eqn-avals-error-p (list i1 f32) :then one :else two))
      (is (%cond-eqn-avals-error-p (list i1 (nb:make-aval '(3) :f32)) :then one :else one))
      (is (%cond-eqn-avals-error-p (list i1) :then one :else one))
      (is (%cond-eqn-avals-error-p (list i1 f32 f32) :then one :else one))
      (is (%cond-eqn-avals-error-p (list i1 f32) :then 1 :else one)))))

(test cond/both-branches-share-the-identical-input-signature
  "両枝のサブグラフの invars は、どちらも eqn の invars（pred を除く）と同じ aval の並びで、
片方しか使わない捕捉値の位置には使われない invar が置かれる。捕捉値の和集合は最初に使った順。"
  (let* ((graph (nb::trace-to-graph
                 (nb:with-tracing (p x y)
                   (nb:cond* p (nb:with-tracing (u) (+ u y)) (nb:with-tracing (u) (* u x)) x))
                 (%cond-avals '(2 3))))
         (eqn (first (nb:graph-eqns graph)))
         (expected (mapcar #'nb:var-aval (rest (nb:eqn-invars eqn)))))
    (is (equal (list (second (nb:graph-invars graph)) (third (nb:graph-invars graph))
                     (second (nb:graph-invars graph)))
               (rest (nb:eqn-invars eqn))))
    (is (equalp expected (mapcar #'nb:var-aval (nb:graph-invars (getf (nb:eqn-params eqn) :then)))))
    (is (equalp expected (mapcar #'nb:var-aval (nb:graph-invars (getf (nb:eqn-params eqn) :else)))))))

(test cond/error-on-zero-outputs
  "枝が値を返さなければ cond-error。"
  (signals nb:cond-error
    (nb::trace-to-graph
     (nb:with-tracing (p x) (nb:cond* p (nb:with-tracing (u) (values)) (nb:with-tracing (u) (values)) x))
     (list (nb:make-aval '() :i1) (nb:make-aval '(2) :f32)))))

(test cond/error-on-branch-arity-mismatch
  "枝の引数の個数が operands の個数と違えば cond-error。"
  (signals nb:cond-error
    (nb::trace-to-graph
     (nb:with-tracing (p x) (nb:cond* p (nb:with-tracing (u v) u) (nb:with-tracing (u) u) x))
     (list (nb:make-aval '() :i1) (nb:make-aval '(2) :f32)))))

(test cond/integer-array-operand-is-lifted
  "operand に整数の配列（:i32）を渡せる。bf16 / f16 の生の配列は cond-error。"
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (p x)
                  (nb:cond* p
                            (nb:with-tracing (u k) (values (+ u 1) k))
                            (nb:with-tracing (u k) (values (- u 1) k))
                            x (make-array 2 :element-type '(signed-byte 32) :initial-element 7)))
                (list (nb:make-aval '() :i1) (nb:make-aval '(2) :f32))))
        (x (make-random-array (make-array-spec '(2) :f32) :seed 9)))
    (multiple-value-bind (a k) (nb:eval-graph graph (%cond-pred 1) x)
      (is (allclose a (reference-add x (make-array 2 :element-type 'single-float :initial-element 1f0)) :dtype :f32))
      (is (equalp k (make-array 2 :element-type '(signed-byte 32) :initial-element 7)))))
  (signals nb:cond-error
    (nb::trace-to-graph
     (nb:with-tracing (p x)
       (nb:cond* p (nb:with-tracing (u k) u) (nb:with-tracing (u k) u)
                 x (make-array 2 :element-type '(unsigned-byte 16))))
     (list (nb:make-aval '() :i1) (nb:make-aval '(2) :f32)))))

(test cond/zero-operands-with-captured-values-only
  "operand が0個でも、枝が閉包で捕まえた外側の値だけで動く。"
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (p x y)
                  (nb:cond* p (nb:with-tracing () (+ x y)) (nb:with-tracing () (* x y))))
                (%cond-avals '(2)))))
    (destructuring-bind (p x y) (%cond-arrays 4 '(2))
      (is (allclose (nb:eval-graph graph p x y) (reference-mul x y) :dtype :f32)))))

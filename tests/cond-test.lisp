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

(test cond/jvp-reports-no-jvp-rule-naming-cond
  "grad / jvp は、#134 まで :cond に jvp ルールが無いことを no-jvp-rule で報告する。"
  (let ((graph (%cond-graph '(2 3))))
    (handler-case (progn (nb::jvp-graph graph) (fail "no-jvp-rule が出なかった"))
      (nb::no-jvp-rule (c) (is (eq :cond (nb::no-jvp-rule-name c)))))))

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
（pred の型・枝の入力 aval・出力 aval の不一致・num-operands）。"
  (let* ((f32 (nb:make-aval '(2) :f32))
         (one (nb::trace-to-graph (nb:with-tracing (u) (+ u u)) (list f32)))
         (two (nb::trace-to-graph (nb:with-tracing (u) (values u u)) (list f32)))
         (i1 (nb:make-aval '() :i1)))
    (is (not (%cond-eqn-avals-error-p (list i1 f32) :then one :else one :num-operands 1)))
    (is (%cond-eqn-avals-error-p (list f32 f32) :then one :else one :num-operands 1))
    (is (%cond-eqn-avals-error-p (list (nb:make-aval '(1) :i1) f32) :then one :else one :num-operands 1))
    (is (%cond-eqn-avals-error-p (list i1 f32) :then one :else two :num-operands 1))
    (is (%cond-eqn-avals-error-p (list i1 (nb:make-aval '(3) :f32)) :then one :else one :num-operands 1))
    (is (%cond-eqn-avals-error-p (list i1 f32) :then one :else one :num-operands 0))
    (is (%cond-eqn-avals-error-p (list i1 f32) :then 1 :else one :num-operands 1))))

(test cond/zero-operands-with-captured-values-only
  "operand が0個でも、枝が閉包で捕まえた外側の値だけで動く。"
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (p x y)
                  (nb:cond* p (nb:with-tracing () (+ x y)) (nb:with-tracing () (* x y))))
                (%cond-avals '(2)))))
    (destructuring-bind (p x y) (%cond-arrays 4 '(2))
      (is (allclose (nb:eval-graph graph p x y) (reference-mul x y) :dtype :f32)))))

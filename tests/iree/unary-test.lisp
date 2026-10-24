;;;; neg / exp / log / tanh / max / min の StableHLO 出力を IREE の local
;;;; backend で実際にコンパイル・実行し、eager 実装と一致することを確かめる
;;;; medium テスト（issue #31 p2）。tests/iree/arith-test.lisp と同じ方針:
;;;; コンパイルは (op, dtype) の組ごとに1回だけ行う。

(in-package #:nabla.iree.tests)

(defmacro def-iree-unary-test (test-name prim-name dtype mlir-op domain)
  "PRIM-NAME（:NEG/:EXP/:LOG/:TANH）・DTYPE（:F32/:BF16）・MLIR-OP から、
shape (4) の1演算モジュールを IREE の local backend でコンパイル・実行し、
PRIM-NAME の eager 実装の結果と allclose :dtype ~A で一致することを確かめる
DEFINE-IREE-TEST を作る。DOMAIN は入力の生成域（log は 0除算・負の値を
避けるため :positive）。"
  `(define-iree-test ,test-name
       ,(format nil "stablehlo.~A（dtype ~(~A~)）を find-backend :iree の
プロトコル経由で実行した結果は、~(~A~) の eager 実装の結果と allclose :dtype
~(~A~) で一致する。" mlir-op dtype prim-name dtype)
     (skip-unless-iree :library :both)
     (let ((aval (nb:make-aval '(4) ,dtype)))
       (with-one-op-module
           ((backend module) (list aval) aval
            (list (format nil "%0 = stablehlo.~A %a0 : ~A" ,mlir-op (nb::tensor-type-string aval))))
         (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                       (lambda (seed)
                         (let* ((spec (make-array-spec '(4) ,dtype))
                                (a (make-random-array spec :seed seed :domain ,domain)))
                           (with-device-arrays ((da (to-device a backend :dtype ,dtype)))
                             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                               (and (equalp (device-array-aval result) aval)
                                    (allclose (to-host result)
                                              (funcall (nb::primitive-eager (nb::find-primitive ,prim-name))
                                                       (list a) (list aval))
                                              :dtype ,dtype))))))
                       :regression-id ,test-name
                       :regression-file (regression-path
                                         ,(format nil "iree-primitives-unary-~(~A~)-~(~A~)" prim-name dtype)
                                         :package "NABLA.IREE.TESTS"))
             ,(format nil "~A/~A: IREE の実行結果が eager 実装と一致しなかった" prim-name dtype))))))

(def-iree-unary-test iree/neg/f32/matches-eager :neg :f32 "negate" :any)
(def-iree-unary-test iree/neg/bf16/matches-eager :neg :bf16 "negate" :any)
(def-iree-unary-test iree/exp/f32/matches-eager :exp :f32 "exponential" :any)
(def-iree-unary-test iree/exp/bf16/matches-eager :exp :bf16 "exponential" :any)
(def-iree-unary-test iree/log/f32/matches-eager :log :f32 "log" :positive)
(def-iree-unary-test iree/log/bf16/matches-eager :log :bf16 "log" :positive)
(def-iree-unary-test iree/tanh/f32/matches-eager :tanh :f32 "tanh" :any)
(def-iree-unary-test iree/tanh/bf16/matches-eager :tanh :bf16 "tanh" :any)
(def-iree-unary-test iree/sqrt/f32/matches-eager :sqrt :f32 "sqrt" :positive)
(def-iree-unary-test iree/sqrt/bf16/matches-eager :sqrt :bf16 "sqrt" :positive)
(def-iree-unary-test iree/sqrt/f64/matches-eager :sqrt :f64 "sqrt" :positive)
(def-iree-unary-test iree/rsqrt/f32/matches-eager :rsqrt :f32 "rsqrt" :positive)
(def-iree-unary-test iree/rsqrt/bf16/matches-eager :rsqrt :bf16 "rsqrt" :positive)
(def-iree-unary-test iree/rsqrt/f64/matches-eager :rsqrt :f64 "rsqrt" :positive)
(def-iree-unary-test iree/log1p/f32/matches-eager :log1p :f32 "log_plus_one" :positive)
(def-iree-unary-test iree/log1p/bf16/matches-eager :log1p :bf16 "log_plus_one" :positive)
(def-iree-unary-test iree/log1p/f64/matches-eager :log1p :f64 "log_plus_one" :positive)
(def-iree-unary-test iree/expm1/f32/matches-eager :expm1 :f32 "exponential_minus_one" :any)
(def-iree-unary-test iree/expm1/bf16/matches-eager :expm1 :bf16 "exponential_minus_one" :any)
(def-iree-unary-test iree/expm1/f64/matches-eager :expm1 :f64 "exponential_minus_one" :any)

(define-iree-test iree/sqrt-rsqrt/matches-jax-fixture
    "sqrt / rsqrt（f32 / f64 / bf16 / f16）を IREE の local backend で実行した結果は、JAX の
フィクスチャ（tests/fixtures/sqrt/）と一致する（NaN・±無限大・±0 は厳密、それ以外は dtype の許容誤差）。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:f32 :f64 :bf16 :f16))
    (let* ((rows (load-sqrt-jax-fixture dtype))
           (x (sqrt-fixture-input-array dtype rows))
           (aval (nb:make-aval (list (length rows)) dtype)))
      (loop for (mlir-op index) in '(("sqrt" 1) ("rsqrt" 2))
            do (with-one-op-module
                   ((backend module) (list aval) aval
                    (list (format nil "%0 = stablehlo.~A %a0 : ~A" mlir-op (nb::tensor-type-string aval))))
                 (with-device-arrays ((da (to-device x backend :dtype dtype)))
                   (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                     (let ((host (to-host result)))
                       (loop for row in rows for i from 0
                             do (is (sqrt-fixture-match-p dtype (aref host i) (nth index row))
                                    "~A/~A: 入力ビット ~D, 期待 ~D, 実際 ~S"
                                    mlir-op dtype (first row) (nth index row) (aref host i)))))))))))

(defmacro def-iree-minmax-test (test-name prim-name dtype mlir-op)
  "PRIM-NAME（:MAX/:MIN）・DTYPE（:F32/:BF16）・MLIR-OP から、shape (4 8) の
1演算モジュールを IREE の local backend でコンパイル・実行し、PRIM-NAME の
eager 実装の結果と allclose :dtype ~A で一致することを確かめる
DEFINE-IREE-TEST を作る。"
  `(define-iree-test ,test-name
       ,(format nil "stablehlo.~A（dtype ~(~A~)）を find-backend :iree の
プロトコル経由で実行した結果は、~(~A~) の eager 実装の結果と allclose :dtype
~(~A~) で一致する。" mlir-op dtype prim-name dtype)
     (skip-unless-iree :library :both)
     (let ((aval (nb:make-aval '(4 8) ,dtype)))
       (with-one-op-module
           ((backend module) (list aval aval) aval
            (list (format nil "%0 = stablehlo.~A %a0, %a1 : ~A" ,mlir-op (nb::tensor-type-string aval))))
         (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                       (lambda (seed)
                         (let* ((spec (make-array-spec '(4 8) ,dtype))
                                (a (make-random-array spec :seed seed))
                                (b (make-random-array spec :seed (1+ seed))))
                           (with-device-arrays ((da (to-device a backend :dtype ,dtype))
                                                (db (to-device b backend :dtype ,dtype)))
                             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                               (and (equalp (device-array-aval result) aval)
                                    (allclose (to-host result)
                                              (funcall (nb::primitive-eager (nb::find-primitive ,prim-name))
                                                       (list a b) (list aval aval))
                                              :dtype ,dtype))))))
                       :regression-id ,test-name
                       :regression-file (regression-path
                                         ,(format nil "iree-primitives-minmax-~(~A~)-~(~A~)" prim-name dtype)
                                         :package "NABLA.IREE.TESTS"))
             ,(format nil "~A/~A: IREE の実行結果が eager 実装と一致しなかった" prim-name dtype))))))

(def-iree-minmax-test iree/max/f32/matches-eager :max :f32 "maximum")
(def-iree-minmax-test iree/max/bf16/matches-eager :max :bf16 "maximum")
(def-iree-minmax-test iree/min/f32/matches-eager :min :f32 "minimum")
(def-iree-minmax-test iree/min/bf16/matches-eager :min :bf16 "minimum")

(defun %log1p-expm1-small-inputs (dtype)
  "小さな |x|（1 + x の丸めで x が消える値）と境界値。"
  (let ((values (if (eq dtype :f64)
                    '(4d-18 -3d-17 1d-12 -1d-10 1d-300 0.5d0 -0.75d0 3d0 -0d0)
                    '(1d-10 -3d-9 1d-6 -1d-5 1d-30 0.5d0 -0.75d0 3d0 -0d0)))
        (type (if (eq dtype :f64) 'double-float 'single-float)))
    (make-array (length values) :element-type type
                                :initial-contents (mapcar (lambda (v) (coerce v type)) values))))

(defun %log1p-expm1-close-p (actual expected dtype)
  "成分ごとに相対誤差 8 ulp（f64 は 1e-15、f32 は 1e-6）。±0 は符号まで一致。"
  (let ((rtol (if (eq dtype :f64) 1d-15 1d-6)))
    (every (lambda (a e)
             (if (zerop e)
                 (and (zerop a) (= (float-sign a) (float-sign e)))
                 (<= (abs (- a e)) (* rtol (abs e)))))
           actual expected)))

(define-iree-test iree/log1p-expm1/keep-the-digits-of-small-inputs-under-jit
    "IREE の local で jit した log1p / expm1 は、小さな |x| でも eager（libm）と相対誤差 8 ulp で一致する
（f32 と f64）。IREE（llvm-cpu）は f64 の log_plus_one / exponential_minus_one を log(1 + x) /
exp(x) - 1 に下ろすので、f64 は emit が Goldberg / Kahan の式を出している（primitives/unary.lisp）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nabla:*compile-cache-directory* nil))
    (dolist (dtype '(:f32 :f64))
      (let ((x (%log1p-expm1-small-inputs dtype)))
        (dolist (entry (list (cons "log1p" (nb:with-tracing (x) (nb:log1p x)))
                             (cons "expm1" (nb:with-tracing (x) (nb:expm1 x)))))
          (let ((jitted (funcall (nb:jit (cdr entry) :backend backend) x))
                (eager (funcall (cdr entry) x)))
            (is (%log1p-expm1-close-p jitted eager dtype) "~A/~A: jit ~S, eager ~S" (car entry) dtype jitted eager)))))))

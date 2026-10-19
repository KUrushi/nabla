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

(def-iree-unary-test iree/expm1/f32/matches-eager :expm1 :f32 "exponential_minus_one" :any)
(def-iree-unary-test iree/expm1/bf16/matches-eager :expm1 :bf16 "exponential_minus_one" :any)
(def-iree-unary-test iree/expm1/f64/matches-eager :expm1 :f64 "exponential_minus_one" :any)
(def-iree-unary-test iree/log1p/f32/matches-eager :log1p :f32 "log_plus_one" :positive)
(def-iree-unary-test iree/log1p/bf16/matches-eager :log1p :bf16 "log_plus_one" :positive)
(def-iree-unary-test iree/log1p/f64/matches-eager :log1p :f64 "log_plus_one" :positive)
(def-iree-unary-test iree/sin/f32/matches-eager :sin :f32 "sine" :any)
(def-iree-unary-test iree/sin/bf16/matches-eager :sin :bf16 "sine" :any)
(def-iree-unary-test iree/sin/f64/matches-eager :sin :f64 "sine" :any)
(def-iree-unary-test iree/cos/f32/matches-eager :cos :f32 "cosine" :any)
(def-iree-unary-test iree/cos/bf16/matches-eager :cos :bf16 "cosine" :any)
(def-iree-unary-test iree/cos/f64/matches-eager :cos :f64 "cosine" :any)

;; IREE（llvm-cpu）と XLA の差（Linux x86_64 の VM で確かめた、issue #223）:
;; - IREE の f32 の sin / cos は多項式で、範囲縮約が |x| に比例して粗い（|x| = 40〜100 で絶対誤差 5e-6 程度。
;;   cos(-67.5) が 1.076e-4 のところ 1.023e-4）。そこで sin / cos の行は |x| <= 10 に限って比べる。
;; - IREE は expm1(-0) / log1p(-0) の f64 と sin(-0) の f32 / bf16 / f16 で +0 を返す（XLA は -0）。
;;   期待値が 0 の行は符号を問わない。
(defparameter *iree-sin-cos-max-abs-input* 10d0)

(defun %iree-fixture-rows (file dtype)
  "FILE の DTYPE の行。sin / cos のファイルは |x| <= *IREE-SIN-COS-MAX-ABS-INPUT*（と非有限）に絞る。"
  (let ((rows (load-unary-jax-fixture file dtype)))
    (if (string= file "jax-sin-cos.lisp")
        (remove-if-not (lambda (row)
                         (let ((x (nabla.tests.support::sqrt-fixture-bits->number dtype (first row))))
                           (or (sb-ext:float-nan-p x) (sb-ext:float-infinity-p x)
                               (<= (abs x) *iree-sin-cos-max-abs-input*))))
                       rows)
        rows)))

(defun %iree-fixture-match-p (dtype actual expected-bits)
  "SQRT-FIXTURE-MATCH-P と同じだが、期待値が 0 なら符号を問わない（上のコメント）。"
  (or (sqrt-fixture-match-p dtype actual expected-bits)
      (let ((e (nabla.tests.support::sqrt-fixture-bits->number dtype expected-bits))
            (a (if (member dtype '(:bf16 :f16))
                   (coerce (nabla::decode-float16 actual dtype) 'double-float)
                   actual)))
        (and (zerop e) (zerop a)))))

(define-iree-test iree/expm1-log1p-sin-cos/matches-jax-fixture
    "expm1 / log1p / sin / cos（f32 / f64 / bf16 / f16）を IREE の local backend で実行した結果は、JAX の
フィクスチャ（tests/fixtures/unary/）と一致する（NaN・±無限大・±0 は厳密、それ以外は dtype の許容誤差）。
sin / cos は |x| ≤ 10 の行だけ、期待値が 0 の行は符号を問わない（上のコメント）。"
  (skip-unless-iree :library :both)
  (loop for (file ops) in '(("jax-expm1-log1p.lisp" (("expm1" "exponential_minus_one") ("log1p" "log_plus_one")))
                            ("jax-sin-cos.lisp" (("sin" "sine") ("cos" "cosine"))))
        do (dolist (dtype '(:f32 :f64 :bf16 :f16))
             (let* ((rows (%iree-fixture-rows file dtype))
                    (x (sqrt-fixture-input-array dtype rows))
                    (aval (nb:make-aval (list (length rows)) dtype)))
               (loop for (name mlir-op) in ops
                     for index from 1
                     do (with-one-op-module
                            ((backend module) (list aval) aval
                             (list (format nil "%0 = stablehlo.~A %a0 : ~A" mlir-op (nb::tensor-type-string aval))))
                          (with-device-arrays ((da (to-device x backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                              (let ((host (to-host result)))
                                (loop for row in rows for i from 0
                                      do (is (%iree-fixture-match-p dtype (aref host i) (nth index row))
                                             "~A/~A: 入力ビット ~D, 期待 ~D, 実際 ~S"
                                             name dtype (first row) (nth index row) (aref host i))))))))))))

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

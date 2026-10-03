;;;; 整数 dtype（:i32 / :u32 / :u64）の IREE での実行（issue #126、medium）。
;;;;
;;;; 整数配列の to-device → to-host の往復と、整数を受け付ける各プリミティブの
;;;; StableHLO を IREE でコンパイル・実行した結果が eager 実装とビット単位で
;;;; 一致すること（オーバーフローで折り返す値を含む。整数は許容誤差なしで
;;;; EQUALP）を確かめる。

(in-package #:nabla.iree.tests)

(define-iree-test device-array/to-host/round-trips-integers-exactly
    "整数 dtype（:i32 / :u32 / :u64）の配列は、どの形状（rank 0..4）でも
to-device → to-host で要素型・値（端の値を含む）が変わらない。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-task)
    (is (check-it (generator (tuple (array-spec :dtypes *integer-dtypes*)
                                    (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (spec-and-seed)
                    (destructuring-bind (spec seed) spec-and-seed
                      (let ((x (make-random-array spec :seed seed)))
                        (with-device-arrays ((y (to-device x device)))
                          (let ((roundtripped (to-host y)))
                            (and (equal (array-element-type roundtripped) (array-element-type x))
                                 (equalp roundtripped x)))))))
                  :regression-id device-array/to-host/round-trips-integers-exactly
                  :regression-file (regression-path "iree-device-array-roundtrip-integers"
                                                    :package "NABLA.IREE.TESTS")))))

(defun %run-graph-on-iree-and-eager (graph arrays)
  "GRAPH を EMIT-STABLEHLO → IREE（local）で実行した結果の配列と、EVAL-GRAPH の
結果の配列を (VALUES iree eager) で返す（出力は1つの graph だけを渡す）。"
  (let* ((backend (nabla:find-backend :iree))
         (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph))))
         (devs (mapcar (lambda (a) (to-device a backend)) arrays)))
    (unwind-protect
         (let ((result (apply #'nabla:backend-invoke backend module "main" devs)))
           (unwind-protect
                (values (to-host result) (apply #'nb:eval-graph graph arrays))
             (release-device-array result)))
      (mapc #'release-device-array devs)
      (nabla:backend-unload backend module))))

(defmacro def-iree-integer-graph-test (test-name docstring (&rest vars) body-form shape)
  "VARS を引数に BODY-FORM を with-tracing でトレースした graph を、整数の3 dtype
それぞれで IREE と eager で実行し、EQUALP で一致することを確かめる。入力はすべて
SHAPE で、端の値（折り返す値）を含むランダムな整数配列。"
  `(define-iree-test ,test-name ,docstring
     (skip-unless-iree :library :both)
     (dolist (dtype *integer-dtypes*)
       (let* ((avals (loop repeat ,(length vars) collect (nb:make-aval ',shape dtype)))
              (graph (nb:trace-to-graph (nb:with-tracing ,vars ,body-form) avals)))
         (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                       (lambda (seed)
                         (let ((arrays (loop for aval in avals for k from 0
                                             collect (make-random-array
                                                      (make-array-spec (nb:aval-shape aval) dtype)
                                                      :seed (+ seed k)))))
                           (multiple-value-bind (iree eager) (%run-graph-on-iree-and-eager graph arrays)
                             (equalp iree eager))))
                       :regression-id ,test-name
                       :regression-file (regression-path
                                         ,(format nil "iree-integer-~(~A~)"
                                                  (substitute #\- #\/ (string test-name)))
                                         :package "NABLA.IREE.TESTS"))
             "~A: IREE の結果が eager と一致しなかった" dtype)))))

(def-iree-integer-graph-test iree/integer/add/matches-eager
    "整数の add（折り返しを含む）が IREE と eager で一致する。"
  (x y) (+ x y) (3 5))

(def-iree-integer-graph-test iree/integer/sub/matches-eager
    "整数の sub が IREE と eager で一致する。" (x y) (- x y) (3 5))

(def-iree-integer-graph-test iree/integer/mul/matches-eager
    "整数の mul が IREE と eager で一致する。" (x y) (* x y) (3 5))

(def-iree-integer-graph-test iree/integer/max/matches-eager
    "整数の max が IREE と eager で一致する（符号付き / 符号なしの大小）。"
  (x y) (max x y) (3 5))

(def-iree-integer-graph-test iree/integer/min/matches-eager
    "整数の min が IREE と eager で一致する。" (x y) (min x y) (3 5))

(def-iree-integer-graph-test iree/integer/neg/matches-eager
    "整数の neg が IREE と eager で一致する（-2^31 は -2^31 のまま）。"
  (x) (- x) (3 5))

(def-iree-integer-graph-test iree/integer/compare-select/matches-eager
    "比較と select（if）が一致する（u64 の上位半分、i32 の負数を含む）。"
  (x y) (if (< x y) x y) (3 5))

(def-iree-integer-graph-test iree/integer/reduce/matches-eager
    "reduce-sum（折り返す）と reduce-max（最小値が初期値）が一致する。"
  (x) (+ (nb:reduce-sum x :axes '(1)) (nb:reduce-max x :axes '(1))) (3 5))

(def-iree-integer-graph-test iree/integer/shape-ops/matches-eager
    "reshape / transpose / broadcast-in-dim が一致する。"
  (x) (nb:broadcast-in-dim (nb:transpose (nb:reshape x '(5 3)) '(1 0)) '(2 3 5) '(1 2)) (3 5))

(def-iree-integer-graph-test iree/integer/literal/matches-eager
    "整数リテラルを含む式（定数の i32 / ui32 / ui64）が一致する。"
  (x) (+ (* x 3) 1) (3 5))

(define-iree-test iree/integer/convert/matches-eager
    "整数 ⇔ 浮動小数点 ⇔ :i1 の convert が IREE と eager で一致する（浮動小数点 →
整数は範囲内の値、整数 → :i1 は 0 以外が 1）。"
  (skip-unless-iree :library :both)
  (flet ((check (from to array)
           (let ((graph (nb:trace-to-graph (nb:with-tracing (x) (nb:convert x to))
                                           (list (nb:make-aval (array-dimensions array) from)))))
             (multiple-value-bind (iree eager) (%run-graph-on-iree-and-eager graph (list array))
               (is (equalp iree eager) "~A -> ~A" from to))))
         (vec (type &rest xs) (make-array (length xs) :element-type type :initial-contents xs)))
    (let ((i32 (vec '(signed-byte 32) -2147483648 -7 0 7 2147483647))
          (u32 (vec '(unsigned-byte 32) 0 1 2147483648 4294967295))
          (u64 (vec '(unsigned-byte 64) 0 5 18446744073709551615))
          (f32 (vec 'single-float -2.7 -0.5 0.0 3.9 1000.0))
          (bit (vec 'bit 0 1 1)))
      (check :i32 :f32 i32)
      (check :i32 :u32 i32)
      (check :i32 :i1 i32)
      (check :u32 :i32 u32)
      (check :u32 :f64 u32)
      (check :u32 :i1 u32)
      (check :u64 :u32 u64)
      (check :u64 :i1 u64)
      (check :f32 :i32 f32)
      (check :f32 :i1 f32)
      (check :i1 :i32 bit)
      (check :i1 :u32 bit)
      (check :i1 :f32 bit))))

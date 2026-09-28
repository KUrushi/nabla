;;;; reduce-sum / reduce-max の medium テスト（issue #31 p6）。
;;;;
;;;; WITH-ONE-OP-MODULE / PRIMITIVE-EAGER-ORACLE
;;;; （tests/iree/primitive-support.lisp）をそのまま再利用する。モジュール本体（%init_0 の宣言 + stablehlo.reduce の
;;;; 2行）は、テスト自身でもう一度書き下すのではなく nb::primitive-emit を
;;;; 呼んで組み立てる（:emit が生成した StableHLO そのものが IREE で
;;;; コンパイル・実行できることを確かめるのが、このテストの目的）。
;;;;
;;;; shape (4 8) を dimensions = [1]（一部の軸）と [0, 1]（全軸、出力が
;;;; rank 0 = tensor<f32> になる）の両方で確かめる（契約のピットフォール
;;;; (2)。全軸を潰すケースは op 対応表の「-> tensor<f32> も可」の注記）。
;;;;
;;;; check-it の :regression-id は（マクロが quote するため）呼び出し
;;;; site に書いたシンボルそのものが使われる。1つの関数に切り出して
;;;; 複数の check-it 呼び出しで共用すると、実行時に違う値を渡しても
;;;; 同じシンボルの regression-cases が全部の呼び出しで共有されてしまう
;;;; ので、4つの define-iree-test はここでは（dot-test.lisp / shape-test.lisp
;;;; と同じ流儀で）あえて別々に書く。"

(in-package #:nabla.iree.tests)

(defun %reduce-body-lines (name in-aval out-aval axes)
  "NAME（:REDUCE-SUM / :REDUCE-MAX）の :EMIT が返す複数行のテキストを、
ONE-OP-MODULE-TEXT に渡せる行のリストに分割する。"
  (uiop:split-string
   (funcall (nb::primitive-emit (nb::find-primitive name)) (list "%a0") (list in-aval) "%0" out-aval :axes axes)
   :separator '(#\Newline)))

(define-iree-test reduce-sum/one-axis-iree-matches-eager
    "shape (4 8)・dimensions = [1] の reduce-sum を IREE local backend で
実行した結果は、eager 実装（host）の結果と f32・bf16 それぞれの許容誤差で
一致する。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:f32 :bf16))
    (let ((in-aval (nb:make-aval '(4 8) dtype))
          (out-aval (nb:make-aval '(4) dtype)))
      (with-one-op-module
          ((backend module) (list in-aval) out-aval
           (%reduce-body-lines :reduce-sum in-aval out-aval '(1)))
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let ((a (make-random-array (make-array-spec '(4 8) dtype) :seed seed)))
                          (with-device-arrays ((da (to-device a backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                              (and (equalp (device-array-aval result) out-aval)
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array
                                                (primitive-eager-oracle :reduce-sum (list a) (list (nb:array-aval a dtype)) :axes '(1))
                                                dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id reduce-sum/one-axis-iree-matches-eager
                      :regression-file (regression-path "iree-reduce-sum-one-axis" :package "NABLA.IREE.TESTS")))))))

(define-iree-test reduce-sum/all-axes-iree-matches-eager
    "shape (4 8)・dimensions = [0, 1]（全軸、出力 tensor<f32>）の
reduce-sum を IREE local backend で実行した結果は、eager 実装（host）の
結果と f32・bf16 それぞれの許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:f32 :bf16))
    (let ((in-aval (nb:make-aval '(4 8) dtype))
          (out-aval (nb:make-aval '() dtype)))
      (with-one-op-module
          ((backend module) (list in-aval) out-aval
           (%reduce-body-lines :reduce-sum in-aval out-aval '(0 1)))
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let ((a (make-random-array (make-array-spec '(4 8) dtype) :seed seed)))
                          (with-device-arrays ((da (to-device a backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                              (and (equalp (device-array-aval result) out-aval)
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array
                                                (primitive-eager-oracle :reduce-sum (list a) (list (nb:array-aval a dtype)) :axes '(0 1))
                                                dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id reduce-sum/all-axes-iree-matches-eager
                      :regression-file (regression-path "iree-reduce-sum-all-axes" :package "NABLA.IREE.TESTS")))))))

(define-iree-test reduce-max/one-axis-iree-matches-eager
    "shape (4 8)・dimensions = [1] の reduce-max を IREE local backend で
実行した結果は、eager 実装（host）の結果と f32・bf16 それぞれの許容誤差で
一致する。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:f32 :bf16))
    (let ((in-aval (nb:make-aval '(4 8) dtype))
          (out-aval (nb:make-aval '(4) dtype)))
      (with-one-op-module
          ((backend module) (list in-aval) out-aval
           (%reduce-body-lines :reduce-max in-aval out-aval '(1)))
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let ((a (make-random-array (make-array-spec '(4 8) dtype) :seed seed)))
                          (with-device-arrays ((da (to-device a backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                              (and (equalp (device-array-aval result) out-aval)
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array
                                                (primitive-eager-oracle :reduce-max (list a) (list (nb:array-aval a dtype)) :axes '(1))
                                                dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id reduce-max/one-axis-iree-matches-eager
                      :regression-file (regression-path "iree-reduce-max-one-axis" :package "NABLA.IREE.TESTS")))))))

(define-iree-test reduce-max/all-axes-iree-matches-eager
    "shape (4 8)・dimensions = [0, 1]（全軸、出力 tensor<f32>）の
reduce-max を IREE local backend で実行した結果は、eager 実装（host）の
結果と f32・bf16 それぞれの許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:f32 :bf16))
    (let ((in-aval (nb:make-aval '(4 8) dtype))
          (out-aval (nb:make-aval '() dtype)))
      (with-one-op-module
          ((backend module) (list in-aval) out-aval
           (%reduce-body-lines :reduce-max in-aval out-aval '(0 1)))
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let ((a (make-random-array (make-array-spec '(4 8) dtype) :seed seed)))
                          (with-device-arrays ((da (to-device a backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                              (and (equalp (device-array-aval result) out-aval)
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array
                                                (primitive-eager-oracle :reduce-max (list a) (list (nb:array-aval a dtype)) :axes '(0 1))
                                                dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id reduce-max/all-axes-iree-matches-eager
                      :regression-file (regression-path "iree-reduce-max-all-axes" :package "NABLA.IREE.TESTS")))))))

(define-iree-test reduce-sum/bf16-f16-axis1024-iree-matches-eager
    "shape (1024) を1本の軸として潰す reduce-sum を bf16・f16 それぞれで
IREE local backend で実行した結果は、eager 実装（host、single-float 累積）
の結果と dtype ごとの許容誤差で一致する（issue #63）。軸長 (4 8) の
ALL-AXES-IREE-MATCHES-EAGER は通っていても、軸長を 1024 まで大きくすると
main では IREE（llvm-cpu）が入力 dtype のまま累積するため一部の seed で
ずれることが分かっている。本体（BODY-LINES）は手書きの文字列ではなく、
プリミティブの実際の :EMIT 出力をそのまま使う。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:bf16 :f16))
    (let ((in-aval (nb:make-aval '(1024) dtype))
          (out-aval (nb:make-aval '() dtype)))
      (with-one-op-module
          ((backend module) (list in-aval) out-aval
           (%reduce-body-lines :reduce-sum in-aval out-aval '(0)))
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let ((a (make-random-array (make-array-spec '(1024) dtype) :seed seed)))
                          (with-device-arrays ((da (to-device a backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                              (and (equalp (device-array-aval result) out-aval)
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array
                                                (primitive-eager-oracle :reduce-sum (list a) (list (nb:array-aval a dtype)) :axes '(0))
                                                dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id reduce-sum/bf16-f16-axis1024-iree-matches-eager
                      :regression-file (regression-path "iree-reduce-sum-axis1024" :package "NABLA.IREE.TESTS")))))))

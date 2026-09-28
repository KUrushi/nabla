;;;; reshape / broadcast-in-dim / transpose の medium テスト（issue #31 p4）。
;;;;
;;;; 各 op につき f32・bf16 でそれぞれ1回だけコンパイルし（1回あたり
;;;; 約350ms、契約 §4 テスト点5）、その中で check-it が複数の seed を
;;;; 試す。shape は正方形にならない・非対称なものを選ぶ（(2 3 4) など）
;;;; ことで、軸の入れ替えミスが数値に現れるようにする（契約のピットフォール
;;;; (6)）。期待値は primitive-eager-oracle（該当プリミティブの eager 実装）
;;;; を host 上でそのまま呼んだ結果にする。

(in-package #:nabla.iree.tests)

(define-iree-test shape/reshape/iree-matches-eager
    "shape (2 3 4) -> (4 3 2) の reshape を IREE local backend で実行した
結果は、eager 実装（host）の結果と f32・bf16 それぞれの許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:f32 :bf16))
    (let ((in-aval (nb:make-aval '(2 3 4) dtype))
          (out-aval (nb:make-aval '(4 3 2) dtype)))
      (with-one-op-module
          ((backend module) (list in-aval) out-aval
           (list (format nil "%0 = stablehlo.reshape %a0 : (~A) -> ~A"
                         (nb::tensor-type-string in-aval) (nb::tensor-type-string out-aval))))
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let* ((a (make-random-array (make-array-spec '(2 3 4) dtype) :seed seed)))
                          (with-device-arrays ((da (to-device a backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                              (and (equalp (device-array-aval result) (nb:make-aval '(4 3 2) dtype))
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array (primitive-eager-oracle :reshape (list a) (list (nb:array-aval a dtype)) :shape '(4 3 2)) dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id shape/reshape/iree-matches-eager
                      :regression-file (regression-path "iree-shape-reshape" :package "NABLA.IREE.TESTS")))))))

(define-iree-test shape/broadcast-in-dim/iree-matches-eager
    "shape (3 4) を dims = [2, 1]（非増加。軸の入れ替えを兼ねる）で
(2 4 3) に broadcast する演算を IREE local backend で実行した結果は、
eager 実装（host）の結果と f32・bf16 それぞれの許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:f32 :bf16))
    (let ((in-aval (nb:make-aval '(3 4) dtype))
          (out-aval (nb:make-aval '(2 4 3) dtype)))
      (with-one-op-module
          ((backend module) (list in-aval) out-aval
           (list (format nil "%0 = stablehlo.broadcast_in_dim %a0, dims = [2, 1] : (~A) -> ~A"
                         (nb::tensor-type-string in-aval) (nb::tensor-type-string out-aval))))
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let* ((a (make-random-array (make-array-spec '(3 4) dtype) :seed seed)))
                          (with-device-arrays ((da (to-device a backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                              (and (equalp (device-array-aval result) (nb:make-aval '(2 4 3) dtype))
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array (primitive-eager-oracle :broadcast-in-dim (list a) (list (nb:array-aval a dtype))
                                                                                     :shape '(2 4 3) :dims '(2 1))
                                                             dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id shape/broadcast-in-dim/iree-matches-eager
                      :regression-file (regression-path "iree-shape-broadcast" :package "NABLA.IREE.TESTS")))))))

(define-iree-test shape/transpose/iree-matches-eager
    "shape (2 3 4) を dims = [2, 0, 1] で transpose する演算を IREE local
backend で実行した結果は、eager 実装（host）の結果と f32・bf16 それぞれの
許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:f32 :bf16))
    (let ((in-aval (nb:make-aval '(2 3 4) dtype))
          (out-aval (nb:make-aval '(4 2 3) dtype)))
      (with-one-op-module
          ((backend module) (list in-aval) out-aval
           (list (format nil "%0 = stablehlo.transpose %a0, dims = [2, 0, 1] : (~A) -> ~A"
                         (nb::tensor-type-string in-aval) (nb::tensor-type-string out-aval))))
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let* ((a (make-random-array (make-array-spec '(2 3 4) dtype) :seed seed)))
                          (with-device-arrays ((da (to-device a backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                              (and (equalp (device-array-aval result) (nb:make-aval '(4 2 3) dtype))
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array (primitive-eager-oracle :transpose (list a) (list (nb:array-aval a dtype)) :perm '(2 0 1))
                                                             dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id shape/transpose/iree-matches-eager
                      :regression-file (regression-path "iree-shape-transpose" :package "NABLA.IREE.TESTS")))))))

(define-iree-test shape/rank0-and-bf16/reshape-and-broadcast-through-iree
    "rank 0 の入出力（reshape の () -> (1 1)、broadcast-in-dim の () ->
(2 3)）を IREE local backend で実行しても、eager 実装（host）と一致する
（bf16 も確かめる）。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:f32 :bf16))
    (let ((scalar-aval (nb:make-aval '() dtype)))
      (with-one-op-module
          ((backend module) (list scalar-aval) (nb:make-aval '(1 1) dtype)
           (list (format nil "%0 = stablehlo.reshape %a0 : (~A) -> ~A"
                         (nb::tensor-type-string scalar-aval) (nb::tensor-type-string (nb:make-aval '(1 1) dtype)))))
        (let* ((a (make-random-array (make-array-spec '() dtype) :seed 7)))
          (with-device-arrays ((da (to-device a backend :dtype dtype)))
            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
              (is (equalp (device-array-aval result) (nb:make-aval '(1 1) dtype)))
              (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                (is (allclose (decode-array (to-host result) dtype)
                              (decode-array (primitive-eager-oracle :reshape (list a) (list (nb:array-aval a dtype)) :shape '(1 1)) dtype)
                              :rtol rtol :atol atol)))))))
      (with-one-op-module
          ((backend module) (list scalar-aval) (nb:make-aval '(2 3) dtype)
           (list (format nil "%0 = stablehlo.broadcast_in_dim %a0, dims = [] : (~A) -> ~A"
                         (nb::tensor-type-string scalar-aval) (nb::tensor-type-string (nb:make-aval '(2 3) dtype)))))
        (let* ((a (make-random-array (make-array-spec '() dtype) :seed 8)))
          (with-device-arrays ((da (to-device a backend :dtype dtype)))
            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
              (is (equalp (device-array-aval result) (nb:make-aval '(2 3) dtype)))
              (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                (is (allclose (decode-array (to-host result) dtype)
                              (decode-array (primitive-eager-oracle :broadcast-in-dim (list a) (list (nb:array-aval a dtype))
                                                                    :shape '(2 3) :dims '())
                                            dtype)
                              :rtol rtol :atol atol))))))))))

;;;; dot-general の medium テスト（issue #31 p5）。
;;;;
;;;; p4 の SHAPE-ONE-OP-MODULE-TEXT / WITH-SHAPE-ONE-OP-MODULE /
;;;; SHAPE-PRIMITIVE-EAGER（tests/iree/shape-primitive-support.lisp）を
;;;; そのまま再利用する（チェーンBの中、同じテストシステムなので DAMP の
;;;; 重複を作らない）。batch なしの (2 3)@(3 4) と、batch 付きの
;;;; (2 3 4)@(2 4 5) の両方を f32・bf16 で確かめ、batching_dims /
;;;; contracting_dims の両方の綴りが IREE でコンパイルできることを
;;;; 検証する（契約のピットフォール(4)）。"

(in-package #:nabla.iree.tests)

(define-iree-test dot-general/no-batch-iree-matches-eager
    "shape (2 3) @ (3 4) -> (2 4) の dot-general（contracting (1)/(0)、
batch なし）を IREE local backend で実行した結果は、eager 実装（host）の
結果と f32・bf16 それぞれの許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:f32 :bf16))
    (let ((lhs-aval (nb:make-aval '(2 3) dtype))
          (rhs-aval (nb:make-aval '(3 4) dtype))
          (out-aval (nb:make-aval '(2 4) dtype)))
      (with-shape-one-op-module (backend module) (list lhs-aval rhs-aval) out-aval
          (list (format nil "%0 = stablehlo.dot_general %a0, %a1, contracting_dims = [1] x [0] : (~A, ~A) -> ~A"
                        (nb::tensor-type-string lhs-aval) (nb::tensor-type-string rhs-aval)
                        (nb::tensor-type-string out-aval)))
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let* ((lhs (make-random-array (make-array-spec '(2 3) dtype) :seed seed))
                               (rhs (make-random-array (make-array-spec '(3 4) dtype) :seed (1+ seed))))
                          (with-device-arrays ((dl (to-device lhs backend :dtype dtype))
                                                (dr (to-device rhs backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" dl dr)))
                              (and (equalp (device-array-aval result) out-aval)
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array
                                                (shape-primitive-eager :dot-general (list lhs rhs)
                                                                       (list (nb:array-aval lhs dtype) (nb:array-aval rhs dtype))
                                                                       :lhs-contracting '(1) :rhs-contracting '(0)
                                                                       :lhs-batch '() :rhs-batch '())
                                                dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id dot-general/no-batch-iree-matches-eager
                      :regression-file (regression-path "iree-dot-general-no-batch" :package "NABLA.IREE.TESTS")))))))

(define-iree-test dot-general/batched-iree-matches-eager
    "shape (2 3 4) @ (2 4 5) -> (2 3 5) の dot-general（batching_dims =
[0] x [0]、contracting_dims = [2] x [1]）を IREE local backend で実行した
結果は、eager 実装（host）の結果と f32・bf16 それぞれの許容誤差で
一致する。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:f32 :bf16))
    (let ((lhs-aval (nb:make-aval '(2 3 4) dtype))
          (rhs-aval (nb:make-aval '(2 4 5) dtype))
          (out-aval (nb:make-aval '(2 3 5) dtype)))
      (with-shape-one-op-module (backend module) (list lhs-aval rhs-aval) out-aval
          (list (format nil "%0 = stablehlo.dot_general %a0, %a1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (~A, ~A) -> ~A"
                        (nb::tensor-type-string lhs-aval) (nb::tensor-type-string rhs-aval)
                        (nb::tensor-type-string out-aval)))
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let* ((lhs (make-random-array (make-array-spec '(2 3 4) dtype) :seed seed))
                               (rhs (make-random-array (make-array-spec '(2 4 5) dtype) :seed (1+ seed))))
                          (with-device-arrays ((dl (to-device lhs backend :dtype dtype))
                                                (dr (to-device rhs backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" dl dr)))
                              (and (equalp (device-array-aval result) out-aval)
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array
                                                (shape-primitive-eager :dot-general (list lhs rhs)
                                                                       (list (nb:array-aval lhs dtype) (nb:array-aval rhs dtype))
                                                                       :lhs-contracting '(2) :rhs-contracting '(1)
                                                                       :lhs-batch '(0) :rhs-batch '(0))
                                                dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id dot-general/batched-iree-matches-eager
                      :regression-file (regression-path "iree-dot-general-batched" :package "NABLA.IREE.TESTS")))))))

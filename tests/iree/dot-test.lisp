;;;; dot-general の medium テスト（issue #31 p5）。
;;;;
;;;; p4 の shape-one-op-module-text / with-shape-one-op-module を再利用する
;;;; （契約のガイダンス: p5 は p4 のヘルパーを使う）。各 op につき f32・
;;;; bf16 でそれぞれ1回だけコンパイルし（契約 §4 テスト点5）、その中で
;;;; check-it が複数の seed を試す。期待値は dot-general の eager 実装
;;;; （host）をそのまま呼んだ結果にする。

(in-package #:nabla.iree.tests)

(defun %dot-eager (arrays in-avals &rest params)
  (apply (nb::primitive-eager (nb::find-primitive :dot-general)) arrays in-avals params))

(define-iree-test dot-general/no-batch-iree-matches-eager
    "shape (2 3) @ (3 4) -> (2 4)（contracting [1] x [0]、batch 無し）を
IREE local backend で実行した結果は、eager 実装（host）の結果と f32・
bf16 それぞれの許容誤差で一致する。"
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
                        (let* ((a (make-random-array (make-array-spec '(2 3) dtype) :seed seed))
                               (b (make-random-array (make-array-spec '(3 4) dtype) :seed (1+ seed))))
                          (with-device-arrays ((da (to-device a backend :dtype dtype))
                                               (db (to-device b backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                              (and (equalp (device-array-aval result) out-aval)
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array (%dot-eager (list a b) (list (nb:array-aval a dtype) (nb:array-aval b dtype))
                                                                          :lhs-contracting '(1) :rhs-contracting '(0)
                                                                          :lhs-batch '() :rhs-batch '())
                                                             dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id dot-general/no-batch-iree-matches-eager
                      :regression-file (regression-path "iree-dot-general-no-batch" :package "NABLA.IREE.TESTS")))))))

(define-iree-test dot-general/batched-iree-matches-eager
    "shape (2 3 4) @ (2 4 5) -> (2 3 5)（batching_dims = [0] x [0]、
contracting_dims = [2] x [1]）を IREE local backend で実行した結果は、
eager 実装（host）の結果と f32・bf16 それぞれの許容誤差で一致する。"
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
                        (let* ((a (make-random-array (make-array-spec '(2 3 4) dtype) :seed seed))
                               (b (make-random-array (make-array-spec '(2 4 5) dtype) :seed (1+ seed))))
                          (with-device-arrays ((da (to-device a backend :dtype dtype))
                                               (db (to-device b backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                              (and (equalp (device-array-aval result) out-aval)
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array (%dot-eager (list a b) (list (nb:array-aval a dtype) (nb:array-aval b dtype))
                                                                          :lhs-contracting '(2) :rhs-contracting '(1)
                                                                          :lhs-batch '(0) :rhs-batch '(0))
                                                             dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id dot-general/batched-iree-matches-eager
                      :regression-file (regression-path "iree-dot-general-batched" :package "NABLA.IREE.TESTS")))))))

(define-iree-test dot-general/bf16-f16-k64-iree-matches-eager
    "shape (4 64) @ (64 4) -> (4 4)（K=64、batch 無し）を bf16・f16 それぞれで
IREE local backend で実行した結果は、eager 実装（host、single-float 累積）
の結果と dtype ごとの許容誤差で一致する（issue #54）。K が小さいテスト
（no-batch-iree-matches-eager）は通っていても、K=64 まで大きくすると
main では IREE が入力 dtype のまま累積するため半分近くの seed で
ずれることが分かっている。本体（BODY-LINES）は手書きの文字列ではなく、
プリミティブの実際の :EMIT 出力をそのまま使う。"
  (skip-unless-iree :library :both)
  (dolist (dtype '(:bf16 :f16))
    (let* ((lhs-aval (nb:make-aval '(4 64) dtype))
           (rhs-aval (nb:make-aval '(64 4) dtype))
           (out-aval (nb:make-aval '(4 4) dtype))
           (emit-text (funcall (nb::primitive-emit (nb::find-primitive :dot-general))
                               '("%a0" "%a1") (list lhs-aval rhs-aval) "%0" out-aval
                               :lhs-contracting '(1) :rhs-contracting '(0)
                               :lhs-batch '() :rhs-batch '()))
           (body-lines (uiop:split-string emit-text :separator '(#\Newline))))
      (with-shape-one-op-module (backend module) (list lhs-aval rhs-aval) out-aval body-lines
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let* ((a (make-random-array (make-array-spec '(4 64) dtype) :seed seed))
                               (b (make-random-array (make-array-spec '(64 4) dtype) :seed (1+ seed))))
                          (with-device-arrays ((da (to-device a backend :dtype dtype))
                                               (db (to-device b backend :dtype dtype)))
                            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                              (and (equalp (device-array-aval result) out-aval)
                                   (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                     (allclose (decode-array (to-host result) dtype)
                                               (decode-array (%dot-eager (list a b) (list (nb:array-aval a dtype) (nb:array-aval b dtype))
                                                                          :lhs-contracting '(1) :rhs-contracting '(0)
                                                                          :lhs-batch '() :rhs-batch '())
                                                             dtype)
                                               :rtol rtol :atol atol)))))))
                      :regression-id dot-general/bf16-f16-k64-iree-matches-eager
                      :regression-file (regression-path "iree-dot-general-k64" :package "NABLA.IREE.TESTS"))
            (format t "~&dot-general k64 ~A: ok~%" dtype))))))

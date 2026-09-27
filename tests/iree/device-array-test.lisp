;;;; device-array の host↔device 往復（issue #7）。

(in-package #:nabla.iree.tests)

(define-iree-test device-array/to-host/round-trips-values-for-f32-and-bf16
    "to-device してから to-host すると、f32 / bf16 のどの形状（rank 0..4、
各次元 1..8）でも元の値が変わらない（allclose、bf16 はビット列そのものが
equalp で一致することも確かめる）。SEED も生成する（固定 :seed 0 だと
形状ごとに値のビットパターンが1通りに固定され、bf16 の inf / NaN /
subnormal / -0 のようなビット列がほとんど生成されないため）。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-task)
    (is (check-it (generator (tuple (array-spec :dtypes '(:f32 :bf16))
                                     (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (spec-and-seed)
                    (destructuring-bind (spec seed) spec-and-seed
                      (let ((x (make-random-array spec :seed seed)))
                        (with-device-arrays ((y (to-device x device :dtype (array-spec-dtype spec))))
                          (let ((roundtripped (to-host y)))
                            (and (allclose roundtripped x :dtype (array-spec-dtype spec))
                                 (or (not (eq (array-spec-dtype spec) :bf16))
                                     (equalp roundtripped x))))))))
                  :regression-id device-array/to-host/round-trips-values-for-f32-and-bf16
                  :regression-file (regression-path "iree-device-array-roundtrip" :package "NABLA.IREE.TESTS")))))

(define-iree-test device-array/device-array-aval/matches-array-aval-and-to-host-dimensions
    "to-device した device-array の aval は array-aval と equalp で一致し、
to-host した結果の array-dimensions は元の shape と一致する。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-task)
    (is (check-it (generator (array-spec :dtypes '(:f32 :bf16)))
                  (lambda (spec)
                    (let ((x (make-random-array spec)))
                      (with-device-arrays ((y (to-device x device :dtype (array-spec-dtype spec))))
                        (and (equalp (device-array-aval y)
                                     (nabla:array-aval x (array-spec-dtype spec)))
                             (equal (array-dimensions (to-host y)) (array-spec-shape spec))))))
                  :regression-id device-array/device-array-aval/matches-array-aval-and-to-host-dimensions
                  :regression-file (regression-path "iree-device-array-aval-matches" :package "NABLA.IREE.TESTS")))))

(define-iree-test device-array/to-device/u16-without-dtype-signals-dtype-mismatch
    "(unsigned-byte 16) の配列を :dtype なしで to-device に渡すと
NABLA:DTYPE-MISMATCH が signal される（bf16 / f16 のどちらか曖昧なため）。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-task)
    (let ((array (make-array '(2 3) :element-type '(unsigned-byte 16) :initial-element 0)))
      (signals nabla:dtype-mismatch (to-device array device)))))

(define-iree-test device-array/to-device/displaced-array-signals-type-error
    "displaced な配列を to-device に渡すと TYPE-ERROR が signal される
（sb-ext:array-storage-vector が使えないので、コピーの前に弾く）。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-task)
    (let* ((backing (make-array 6 :element-type 'single-float :initial-element 0.0f0))
           (displaced (make-array '(2 3) :element-type 'single-float :displaced-to backing)))
      (signals type-error (to-device displaced device)))))

(define-iree-test device-array/release-device-array/idempotent-and-blocks-to-host
    "release-device-array は idempotent（二重に呼んでも何も起きない）で、
device-array-released-p はその前後で切り替わる。release 後の device-array
を to-host に渡すと IREE-OBJECT-RELEASED（kind :device-array）が signal
される。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-task)
    (let* ((x (make-random-array (make-array-spec '(2 3) :f32)))
           (y (to-device x device)))
      (is (not (device-array-released-p y)))
      (release-device-array y)
      (is (device-array-released-p y))
      (release-device-array y)
      (is (device-array-released-p y))
      (handler-case
          (progn
            (to-host y)
            (fiveam:fail "released device-array should signal iree-object-released"))
        (iree-object-released (condition)
          (is (eq :device-array (iree-object-released-kind condition))))))))

(define-iree-test device-array/to-host/survives-release-device-on-wrapper
    "device-array は生成時に device を retain しているので、その device
オブジェクト自身を release-device した後でも、既存の device-array の
to-host はまだ正しい値を返し、release-device-array もクラッシュしない
（W3 契約の fact 5）。"
  (skip-unless-iree :library :runtime)
  (let* ((device (make-device :local-task))
         (x (make-random-array (make-array-spec '(2 3) :f32)))
         (y (to-device x device)))
    (unwind-protect
         (progn
           (release-device device)
           (is (allclose (to-host y) x :dtype :f32)))
      ;; to-host が失敗しても y と、y が retain している device を必ず
      ;; 解放する（#11 で finalizer が入るまでは、ここで解放し忘れると
      ;; このテストランの間ずっとリークする）。
      (release-device-array y))
    (is (device-array-released-p y))))

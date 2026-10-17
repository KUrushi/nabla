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
    (is (check-it (generator (array-spec :dtypes '(:f32 :bf16 :f64 :i1)))
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

(define-iree-test device-array/to-host/round-trips-f64-and-i1-exactly
    "to-device してから to-host すると、f64 / i1 のどの形状（rank 0..4、
各次元 1..8）でも元の配列と equalp で一致する（issue #72）。f64 は丸めが
入らないのでビット単位で一致し、i1 は BIT 配列（ビット詰め）と IREE の
1要素1バイト（IREE_HAL_ELEMENT_TYPE_BOOL_8）の間の詰め直しが要素の位置を
ずらさないこと（8 や 64 の倍数でない要素数を含む）を確かめる。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-task)
    (is (check-it (generator (tuple (array-spec :dtypes '(:f64 :i1))
                                     (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (spec-and-seed)
                    (destructuring-bind (spec seed) spec-and-seed
                      (let ((x (make-random-array spec :seed seed)))
                        (with-device-arrays ((y (to-device x device)))
                          (let ((roundtripped (to-host y)))
                            (and (equal (array-element-type roundtripped) (array-element-type x))
                                 (equalp roundtripped x)))))))
                  :regression-id device-array/to-host/round-trips-f64-and-i1-exactly
                  :regression-file (regression-path "iree-device-array-roundtrip-f64-i1" :package "NABLA.IREE.TESTS")))))

(define-iree-test device-array/to-device/zero-size-f64-and-i1-round-trip
    "要素数0の f64 / i1 の配列（shape (0) と (2 0 3)）も to-device → to-host
で同じ shape・要素型の配列に戻る（issue #72。i1 の詰め直しが空の配列で
範囲外を読み書きしない）。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-task)
    (dolist (x (list (make-array '(0) :element-type 'double-float)
                     (make-array '(2 0 3) :element-type 'bit)))
      (with-device-arrays ((y (to-device x device)))
        (let ((roundtripped (to-host y)))
          (is (equal (array-dimensions x) (array-dimensions roundtripped)))
          (is (equal (array-element-type x) (array-element-type roundtripped))))))))

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

(define-iree-test device-array/to-host/staged-read-matches-direct-read
    "ホストから見えないバッファ（CUDA の device-local）の to-host は、デバイスごとに
1つの作業用バッファを使い回して読む（issue #190）。local のバッファはホストから
見えるのでこの経路を通らないため、内部の %READ-VIA-STAGING を
:EVEN-IF-HOST-VISIBLE で直接呼んで確かめる（cuda での実際の経路は
device-array/to-host/cuda-round-trips-through-staging）。同じ device で大きさの
違う配列を続けて読み（作業用バッファが伸びる・使い回される両方を通す）、どれも
to-host と同じ値（equalp）になる。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-task)
    (is (check-it (generator (tuple (list (array-spec :dtypes '(:f32 :bf16 :f64 :i1 :i32)))
                                     (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (specs-and-seed)
                    (destructuring-bind (specs seed) specs-and-seed
                      (every (lambda (spec)
                               (let ((x (make-random-array spec :seed seed)))
                                 (with-device-arrays ((y (to-device x device :dtype (array-spec-dtype spec))))
                                   (equalp (nabla.iree::%read-via-staging y :even-if-host-visible t)
                                           (to-host y)))))
                             specs)))
                  :regression-id device-array/to-host/staged-read-matches-direct-read
                  :regression-file (regression-path "iree-device-array-staged-read" :package "NABLA.IREE.TESTS")))))

(define-iree-test device-array/to-host/staged-read-survives-release-device
    "作業用バッファで読んだ後に release-device しても、まだ生きている device-array は
to-host で同じ値を返し、その後の release-device-array も落ちない（作業用バッファは
release-device が device より先に解放する。issue #190）。"
  (skip-unless-iree :library :runtime)
  (let* ((device (make-device :local-task))
         (x (make-random-array (make-array-spec '(3 5) :f32) :seed 7))
         (y (to-device x device)))
    (unwind-protect
         (let ((staged (nabla.iree::%read-via-staging y :even-if-host-visible t)))
           (release-device device)
           (is (equalp staged x))
           (is (equalp (to-host y) x)))
      (release-device-array y)
      (release-device device))))

(define-iree-test/large device-array/to-host/cuda-round-trips-through-staging
    "cuda では to-host が作業用バッファの経路を通り（issue #190）、大きさや dtype の
違う配列を同じ device で続けて読んでも、to-device した値がそのまま戻る（equalp）。"
  (skip-unless-iree :library :runtime)
  (skip-unless-cuda)
  (with-device (device :cuda)
    (is (check-it (generator (tuple (list (array-spec :dtypes '(:f32 :bf16 :f64 :i1 :i32)))
                                     (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (specs-and-seed)
                    (destructuring-bind (specs seed) specs-and-seed
                      (every (lambda (spec)
                               (let ((x (make-random-array spec :seed seed)))
                                 (with-device-arrays ((y (to-device x device :dtype (array-spec-dtype spec))))
                                   (equalp (to-host y) x))))
                             specs)))
                  :regression-id device-array/to-host/cuda-round-trips-through-staging
                  :regression-file (regression-path "iree-device-array-cuda-staged" :package "NABLA.IREE.TESTS")))
    (is (plusp (nabla.iree::%device-staging-size device)))))

(defun %live-allocator-bytes (device)
  "DEVICE の allocator が確保して、まだ解放していないバイト数（host と device の合計）。"
  (let ((s (device-allocator-statistics device)))
    (+ (- (allocator-statistics-host-bytes-allocated s) (allocator-statistics-host-bytes-freed s))
       (- (allocator-statistics-device-bytes-allocated s) (allocator-statistics-device-bytes-freed s)))))

(define-iree-test device-array/to-host/staging-buffer-is-reused-not-leaked
    "作業用バッファは device ごとに1つだけ持つ（issue #190）。大きさの違う配列を
続けて作業用バッファの経路で読み、device-array をすべて解放した後に残る確保は
作業用バッファ1つ分だけで、その大きさは読んだ最大のバイト数以上、その2倍未満
（ただし 4 KiB 以上）に収まる（伸ばすたびに古いバッファを解放し、
必要以上に大きく取らない）。"
  (skip-unless-iree :library :runtime)
  (is (check-it (generator (list (array-spec :dtypes '(:f32 :f64 :i32)) :min-length 1))
                (lambda (specs)
                  (with-device (device :local-sync)
                    (let ((baseline (%live-allocator-bytes device)))
                      (dolist (spec specs)
                        (with-device-arrays ((y (to-device (make-random-array spec) device
                                                           :dtype (array-spec-dtype spec))))
                          (nabla.iree::%read-via-staging y :even-if-host-visible t)))
                      (let ((live (- (%live-allocator-bytes device) baseline))
                            (largest (reduce #'max specs
                                             :key (lambda (spec)
                                                    (nabla:aval-byte-length
                                                     (nabla:make-aval (array-spec-shape spec)
                                                                      (array-spec-dtype spec)))))))
                        (and (<= largest live)
                             (<= live (max 4096 (1- (* 2 largest)))))))))
                :regression-id device-array/to-host/staging-buffer-is-reused-not-leaked
                :regression-file (regression-path "iree-device-array-staging-reuse" :package "NABLA.IREE.TESTS"))))

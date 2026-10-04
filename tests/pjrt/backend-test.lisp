;;;; nabla:backend プロトコルの前半（find-backend / to-device / to-host /
;;;; device-array-aval）を、PJRT の CPU プラグインで確かめるテスト（issue #85）。
;;;; tests/iree/backend-test.lisp と同じ性質を PJRT でも成り立たせる。
;;;;
;;;; 生の CFFI（PJRT の呼び出しの薄い包み）は mutation testing の対象外
;;;; （tools/mutate/README.md）。ここのテストは、その CFFI が正しく動くことの
;;;; 疎通と、寿命・解放の約束を確かめる。

(in-package #:nabla.pjrt.tests)

(defun %pjrt-backend ()
  (nabla:find-backend :pjrt))

(defun %roundtrip-equal-p (x roundtripped dtype)
  "ROUNDTRIPPED が X と同じ要素型・形で、値が一致するか。bf16 / f16 は
ビット列そのものが equalp で一致することを要求する（NaN のビット列も含めて
保たれること）。f32 / f64 は allclose。"
  (and (equal (array-element-type x) (array-element-type roundtripped))
       (equal (array-dimensions x) (array-dimensions roundtripped))
       (if (member dtype '(:bf16 :f16))
           (equalp x roundtripped)
           (allclose roundtripped x :dtype dtype))))

(define-pjrt-test backend/find-backend/returns-the-same-pjrt-backend
  "(find-backend :pjrt) は pjrt-backend 型のオブジェクトを返し、2回呼んでも
同じ（eq）オブジェクトを返す。プラットフォーム名は \"cpu\"。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (nabla:find-backend :pjrt)))
    (is (typep backend 'pjrt-backend))
    (is (eq backend (nabla:find-backend :pjrt)))
    (is (eq :cpu (nabla:backend-target backend)))
    (is (string= "cpu" (pjrt-backend-platform-name backend)))))

(define-pjrt-test backend/to-host/round-trips-values
  "to-device してから to-host すると、f32 / bf16 / f16 / f64 のどの形状
（rank 0..4、各次元 1..8）でも元の値が変わらない。bf16 / f16 はビット列が
equalp で一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (%pjrt-backend)))
    (is (check-it (generator (tuple (array-spec :dtypes '(:f32 :bf16 :f16 :f64))
                                    (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (spec-and-seed)
                    (destructuring-bind (spec seed) spec-and-seed
                      (let ((x (make-random-array spec :seed seed))
                            (dtype (array-spec-dtype spec)))
                        (with-pjrt-arrays ((y (nabla:to-device x backend :dtype dtype)))
                          (%roundtrip-equal-p x (nabla:to-host y) dtype)))))
                  :regression-id backend/to-host/round-trips-values
                  :regression-file (regression-path "pjrt-backend-roundtrip"
                                                    :package "NABLA.PJRT.TESTS")))))

(define-pjrt-test backend/device-array-aval/matches-array-aval
  "to-device した device-array の aval は nabla:array-aval と equalp で一致し、
to-host した結果の array-dimensions は元の shape と一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (%pjrt-backend)))
    (is (check-it (generator (array-spec :dtypes '(:f32 :bf16 :f16 :f64)))
                  (lambda (spec)
                    (let ((x (make-random-array spec))
                          (dtype (array-spec-dtype spec)))
                      (with-pjrt-arrays ((y (nabla:to-device x backend :dtype dtype)))
                        (and (equalp (nabla:device-array-aval y) (nabla:array-aval x dtype))
                             (equal (array-dimensions (nabla:to-host y)) (array-spec-shape spec))))))
                  :regression-id backend/device-array-aval/matches-array-aval
                  :regression-file (regression-path "pjrt-backend-aval"
                                                    :package "NABLA.PJRT.TESTS")))))

(define-pjrt-test backend/to-device/example-2x2
  "疎通の例: #2A((1.0 2.0) (3.0 4.0)) の往復で元の配列に戻る。"
  (skip-unless-pjrt :kind :cpu)
  (let ((x (make-array '(2 2) :element-type 'single-float
                              :initial-contents '((1.0 2.0) (3.0 4.0)))))
    (with-pjrt-arrays ((y (nabla:to-device x (%pjrt-backend))))
      (is (equalp x (nabla:to-host y))))))

(define-pjrt-test backend/to-device/zero-size-arrays-round-trip
  "要素数0の配列（shape (0) と (2 0 3)）も、同じ shape・要素型で往復する。"
  (skip-unless-pjrt :kind :cpu)
  (dolist (x (list (make-array '(0) :element-type 'single-float)
                   (make-array '(2 0 3) :element-type 'double-float)))
    (with-pjrt-arrays ((y (nabla:to-device x (%pjrt-backend))))
      (let ((roundtripped (nabla:to-host y)))
        (is (equal (array-dimensions x) (array-dimensions roundtripped)))
        (is (equal (array-element-type x) (array-element-type roundtripped)))))))

(define-pjrt-test backend/to-device/rejects-bad-arguments
  "(unsigned-byte 16) を :dtype なしで渡すと nabla:dtype-mismatch、
simple-array でない配列は type-error。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (%pjrt-backend)))
    (signals nabla:dtype-mismatch
      (nabla:to-device (make-array '(2 3) :element-type '(unsigned-byte 16) :initial-element 0)
                       backend))
    (signals type-error
      (nabla:to-device (make-array '(2 2) :element-type 'single-float :adjustable t
                                          :initial-element 0.0)
                       backend))))

(define-pjrt-test backend/make-backend/without-plugin-signals-backend-not-available
  "NABLA_PJRT_HOME がプラグインの無い場所を指すとき、make-backend は
nabla:backend-not-available（kind :pjrt）を signal する。環境を戻した後の
find-backend は成功する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((old (sb-ext:posix-getenv "NABLA_PJRT_HOME")))
    (unwind-protect
         (progn
           (sb-posix:setenv "NABLA_PJRT_HOME" "/nonexistent-pjrt-home" 1)
           (is (not (pjrt-available-p)))
           (handler-case (nabla:make-backend :pjrt)
             (nabla:backend-not-available (condition)
               (is (eq :pjrt (nabla:backend-not-available-kind condition))))
             (:no-error (&rest values)
               (declare (ignore values))
               (fail "make-backend :pjrt should have signalled")))
           ;; find-backend は作成済みのインスタンスを返すので、ここでは試さない。
           )
      (if old
          (sb-posix:setenv "NABLA_PJRT_HOME" old 1)
          (sb-posix:unsetenv "NABLA_PJRT_HOME")))
    (is (typep (nabla:find-backend :pjrt) 'pjrt-backend))))

(define-pjrt-test backend/make-backend/device-index-out-of-range-is-an-error
  "addressable なデバイスの数以上の :device-index は error。"
  (skip-unless-pjrt :kind :cpu)
  (signals error (nabla:make-backend :pjrt :device-index 1000000))
  (signals type-error (nabla:make-backend :pjrt :target :tpu)))

(define-pjrt-test backend/pjrt-error/carries-the-plugin-message
  "PJRT が PJRT_Error を返したとき、メッセージを取り出して pjrt-error
（nabla:backend-error の子）になる。小さすぎる dst_size で
PJRT_Buffer_ToHostBuffer を呼んで、実際にエラーを起こす（メッセージに
必要なバイト数 16 が入る）。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((backend (%pjrt-backend))
         (client (nabla.pjrt::%pjrt-backend-client backend))
         (api (nabla.pjrt::%pjrt-client-api client)))
    (with-pjrt-arrays ((y (nabla:to-device (make-array 4 :element-type 'single-float
                                                         :initial-element 1.0)
                                           backend)))
      (cffi:with-foreign-object (dst :uint8 64)
        (handler-case
            (nabla.pjrt::%pjrt-call (api "PJRT_Buffer_ToHostBuffer" args
                                         (:struct nabla.pjrt::%buffer-to-host-buffer-args)
                                         (nabla.pjrt::src (nabla.pjrt::%device-array-pointer y))
                                         (nabla.pjrt::dst dst)
                                         (nabla.pjrt::dst-size 1))
              (fail "a too small dst_size should have been rejected"))
          (nabla:backend-error (condition)
            (is (typep condition 'pjrt-error))
            ;; 固定したプラグインのエラー文言に依存（"must be >= 16"）。
            (is (search "16" (pjrt-error-message condition)))
            (is (string= "PJRT_Buffer_ToHostBuffer" (pjrt-error-context condition)))))))))

(define-pjrt-test device-array/release/is-idempotent-and-survives-gc
  "release-device-array は2回呼んでもエラーにならず、解放済みの device-array の
to-host は pjrt-object-released。その後 GC と finalizer を走らせても落ちない
（finalizer は tg:cancel-finalization で取り消されているので二重解放しない）。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (%pjrt-backend)))
    (dotimes (i 50)
      (let ((y (nabla:to-device (make-array '(4) :element-type 'single-float :initial-element 1.0)
                                backend)))
        (release-device-array y)
        (is (device-array-released-p y))
        (finishes (release-device-array y))
        (when (zerop i)
          (signals pjrt-object-released (nabla:to-host y)))))
    (finishes (gc-and-run-finalizers))
    (finishes (gc-and-run-finalizers))))

(defun %live-buffers (backend)
  (nabla.pjrt::client-state-live-buffers
   (nabla.pjrt::%pjrt-client-state (nabla.pjrt::%pjrt-backend-client backend))))

(defun %drop-arrays (backend count)
  "COUNT 個の device-array を作って、明示的に解放せずに捨てる。呼び出し元の
スタックに参照を残さないよう、別の関数にしてある（notinline）。"
  (declare (notinline nabla:to-device))
  (dotimes (i count)
    (nabla:to-device (make-array '(2 2) :element-type 'single-float :initial-element 1.0)
                     backend)))

(define-pjrt-test device-array/finalizer/frees-dropped-buffers
  "device-array を作っては参照を捨てる（release-device-array しない）ループの
後で GC と finalizer を走らせると、生きているバッファの数（client-state の
カウンタ）が元に近い値に戻る。保守的なスタックルートで少数は残りうるので、
ちょうど 0 ではなく、作った数のごく一部以下であることを確かめる。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((backend (%pjrt-backend))
         (baseline (%live-buffers backend))
         (count 2000))
    (%drop-arrays backend count)
    (is (>= (- (%live-buffers backend) baseline) 1))
    (dotimes (i 3) (gc-and-run-finalizers))
    (is (<= (- (%live-buffers backend) baseline) 20)
        "~D buffers still alive after GC" (- (%live-buffers backend) baseline))))

(defun %make-array-on-private-backend ()
  "専用のクライアント（find-backend と共有しない）を作り、device-array を1つ
作って、(values device-array client-state) を返す。backend と client への
参照は、この関数のスタックフレームとともに消える。"
  (declare (notinline nabla:make-backend nabla:to-device))
  (let* ((backend (nabla:make-backend :pjrt))
         (array (nabla:to-device (make-array '(2) :element-type 'single-float
                                                  :initial-contents '(1.0 2.0))
                                 backend)))
    (values array
            (nabla.pjrt::%pjrt-client-state (nabla.pjrt::%pjrt-backend-client backend)))))

(define-pjrt-test device-array/outlives-its-backend
  "クライアントの所有者が消えた（owner-gone）後でも、生きている device-array の
to-host は動き続け（クライアントはバッファより先に破棄されない）、
最後のバッファを解放した時点で初めてクライアントが破棄される。
所有者の finalizer は GC 任せで不確定なので、%client-state-owner-gone を
直接呼んで同じ状態を作る。"
  (skip-unless-pjrt :kind :cpu)
  (multiple-value-bind (array state) (%make-array-on-private-backend)
    (nabla.pjrt::%client-state-owner-gone state)
    (is (not (nabla.pjrt::client-state-destroyed-p state))
        "the client was destroyed while a buffer was still alive")
    (is (equalp #(1.0 2.0) (nabla:to-host array)))
    (release-device-array array)
    (is (nabla.pjrt::client-state-destroyed-p state))))

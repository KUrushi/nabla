;;;; nabla.iree のランタイム C API バインディングのテスト（issue #6）。
;;;;
;;;; local（CPU）のドライバ・デバイス・セッション・呼び出し・buffer view を
;;;; 通しで確かめる。CUDA を使うテストは tests/iree/runtime-cuda-test.lisp
;;;; （:nabla.large）に分けてある。

(in-package #:nabla.iree.tests)

(define-iree-test runtime/make-device/local-drivers-create-and-release-in-a-loop
    "local-task と local-sync のそれぞれについて、device の作成と解放を
50回繰り返しても失敗しない。released-p は解放後に真になり、二重解放しても
何も起きない（idempotent）。"
  (skip-unless-iree :library :runtime)
  (dolist (driver '(:local-task :local-sync))
    (dotimes (i 50)
      (let ((device (make-device driver)))
        (is (not (device-released-p device)))
        (is (eq driver (device-driver device)))
        (release-device device)
        (is (device-released-p device))
        ;; 二重解放は何もしない。
        (release-device device)
        (is (device-released-p device))))))

(define-iree-test runtime/make-device/unknown-driver-signals-not-found
    "(driver-names) に無いドライバ名を渡すと、IREE-STATUS-ERROR（code
:not-found）が signal される。"
  (skip-unless-iree :library :runtime)
  (let ((known (driver-names)))
    (is (check-it (generator (map (lambda (suffix)
                                     (concatenate 'string "nabla-unknown-driver-" suffix))
                                   (string)))
                  (lambda (suffix)
                    (let ((name (concatenate 'string "nabla-unknown-driver-" suffix)))
                      (or (member name known :test #'string=)
                          (handler-case
                              (progn (release-device (make-device name))
                                     nil)
                            (iree-status-error (condition)
                              (eq :not-found (iree-status-error-code condition)))))))
                  :regression-id runtime/make-device/unknown-driver-signals-not-found
                  :regression-file (regression-path "iree-runtime-unknown-driver" :package "NABLA.IREE.TESTS")))))

(define-iree-test runtime/driver-names/contains-local-drivers
    "driver-names の結果に local-task と local-sync が両方含まれる。"
  (skip-unless-iree :library :runtime)
  (let ((names (driver-names)))
    (is (member "local-task" names :test #'string=))
    (is (member "local-sync" names :test #'string=))))

(define-iree-test runtime/session/loads-vmfb-and-lists-main
    "compile-stablehlo（#5）でコンパイルした matmul フィクスチャの vmfb を
session-append-module でロードすると、session-function-names に \"main\" が
含まれ、session-lookup-function \"module.main\" は linkage 2（EXPORT）・
ordinal 0 で見つかり、存在しない \"module.nope\" は :not-found になる。"
  (skip-unless-iree :library :both)
  (with-device (device :local)
    (with-session (session device)
      (let ((bytes (compile-stablehlo (stablehlo-fixture "matmul"))))
        (session-append-module session bytes)
        (is (member "main" (session-function-names session) :test #'string=))
        (let ((function (session-lookup-function session "module.main")))
          (is (= 2 (vm-function-linkage function)))
          (is (= 0 (vm-function-ordinal function))))
        (handler-case
            (progn (session-lookup-function session "module.nope")
                   (fiveam:fail "module.nope should have signalled iree-status-error"))
          (iree-status-error (condition)
            (is (eq :not-found (iree-status-error-code condition)))))))))

(define-iree-test runtime/session/corrupt-module-signals-status-error
    "壊れた（vmfb として無意味な）バイト列を session-append-module に渡すと
IREE-STATUS-ERROR が signal される。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local)
    (with-session (session device)
      (signals iree-status-error
        (session-append-module
         session (make-array 64 :element-type '(unsigned-byte 8) :initial-element #xFF))))))

(define-iree-test runtime/session/append-module-from-file-matches-in-memory
    "iree-compile（CLI、#5 とは別経路）が書いた vmfb ファイルを
session-append-module-from-file でロードした場合も、compile-stablehlo が
返すバイト列を session-append-module でロードした場合と同じく \"main\" が
呼べる（in-memory 経路とファイル経路の交差確認）。"
  (skip-unless-iree :library :both)
  (let ((iree-compile (merge-pathnames "bin/iree-compile" (nabla.iree::iree-home))))
    (unless (probe-file iree-compile)
      (fiveam:skip "~A が無いので iree-compile 経由の交差確認をスキップする" iree-compile)
      (return-from iree-test))
    (let ((vmfb-path (merge-pathnames
                       (format nil "nabla-iree-runtime-test-~A.vmfb" (random 1000000))
                       (uiop:temporary-directory)))
          (mlir-path (merge-pathnames
                      (format nil "nabla-iree-runtime-test-~A.mlir" (random 1000000))
                      (uiop:temporary-directory))))
      (unwind-protect
           (progn
             (with-open-file (stream mlir-path :direction :output :if-exists :supersede)
               (write-string (stablehlo-fixture "matmul") stream))
             (multiple-value-bind (output error-output exit-code)
                 (uiop:run-program
                  (list (namestring iree-compile)
                        "--iree-input-type=stablehlo"
                        "--iree-hal-target-device=local"
                        "--iree-hal-local-target-device-backends=llvm-cpu"
                        "--iree-llvmcpu-target-cpu=host"
                        (namestring mlir-path)
                        "-o" (namestring vmfb-path))
                  :output '(:string) :error-output '(:string) :ignore-error-status t)
               (declare (ignore output))
               (is (zerop exit-code) "iree-compile failed: ~A" error-output))
             (with-device (device :local)
               (with-session (session device)
                 (session-append-module-from-file session vmfb-path)
                 (is (member "main" (session-function-names session) :test #'string=)))))
        (ignore-errors (delete-file vmfb-path))
        (ignore-errors (delete-file mlir-path))))))

(defun %f32-octets (values)
  "VALUES（single-float のリスト）を、リトルエンディアン f32 のバイト列
（(unsigned-byte 8) ベクタ）にする。"
  (let* ((count (length values))
         (bytes (make-array (* 4 count) :element-type '(unsigned-byte 8)))
         (floats (make-array count :element-type 'single-float :initial-contents values)))
    (sb-sys:with-pinned-objects (floats bytes)
      (cffi:foreign-funcall "memcpy"
                             :pointer (sb-sys:vector-sap bytes)
                             :pointer (sb-sys:vector-sap floats)
                             :size (* 4 count)
                             :pointer))
    bytes))

(defun %octets-to-f32-array (octets dimensions)
  "OCTETS（リトルエンディアン f32 のバイト列）を DIMENSIONS の形の
single-float の配列にする。"
  (let* ((count (reduce #'* dimensions))
         (floats (make-array count :element-type 'single-float)))
    (sb-sys:with-pinned-objects (octets floats)
      (cffi:foreign-funcall "memcpy"
                             :pointer (sb-sys:vector-sap floats)
                             :pointer (sb-sys:vector-sap octets)
                             :size (* 4 count)
                             :pointer))
    (make-array dimensions :element-type 'single-float
                           :displaced-to floats)))

(define-iree-test runtime/call/matmul-fixture-computes-expected
    "matmul フィクスチャに #(1 2 3 4 5 6) (shape 2x3) と
#(7 8 9 10 11 12) (shape 3x2) を渡して呼び出すと、shape (2 2) の
#(58 64 139 154) が返る。"
  (skip-unless-iree :library :both)
  (with-device (device :local)
    (with-session (session device)
      (session-append-module session (compile-stablehlo (stablehlo-fixture "matmul")))
      (let* ((a-bytes (%f32-octets '(1.0f0 2.0f0 3.0f0 4.0f0 5.0f0 6.0f0)))
             (b-bytes (%f32-octets '(7.0f0 8.0f0 9.0f0 10.0f0 11.0f0 12.0f0)))
             result)
        (sb-sys:with-pinned-objects (a-bytes b-bytes)
          (let ((bv-a (buffer-view-allocate-copy device '(2 3) :f32
                                                  (sb-sys:vector-sap a-bytes) (length a-bytes)))
                (bv-b (buffer-view-allocate-copy device '(3 2) :f32
                                                  (sb-sys:vector-sap b-bytes) (length b-bytes))))
            (unwind-protect
                 (with-call (call session "module.main")
                   (call-push-buffer-view call bv-a)
                   (call-push-buffer-view call bv-b)
                   (call-invoke call)
                   (setf result (call-pop-buffer-view call)))
              (buffer-view-release bv-a)
              (buffer-view-release bv-b))))
        (unwind-protect
             (progn
               (is (equal '(2 2) (buffer-view-shape result)))
               (is (eq :f32 (buffer-view-element-type result)))
               (let ((octets (make-array (buffer-view-byte-length result)
                                          :element-type '(unsigned-byte 8))))
                 (buffer-view-read-into device result octets)
                 (is (allclose (%octets-to-f32-array octets '(2 2))
                                (make-array '(2 2) :element-type 'single-float
                                                    :initial-contents '((58.0f0 64.0f0) (139.0f0 154.0f0)))
                                :dtype :f32))))
          (buffer-view-release result))))))

(define-iree-test runtime/buffer-view/element-type-table-matches-runtime
    "matmul フィクスチャの結果 buffer view の要素型は :f32 に decode される
（buffer-view.h の IREE_HAL_ELEMENT_TYPE_FLOAT_32 から手計算した定数の検算）。"
  (skip-unless-iree :library :both)
  (with-device (device :local)
    (with-session (session device)
      (session-append-module session (compile-stablehlo (stablehlo-fixture "matmul")))
      (let* ((a-bytes (%f32-octets '(1.0f0 2.0f0 3.0f0 4.0f0 5.0f0 6.0f0)))
             (b-bytes (%f32-octets '(7.0f0 8.0f0 9.0f0 10.0f0 11.0f0 12.0f0))))
        (sb-sys:with-pinned-objects (a-bytes b-bytes)
          (let ((bv-a (buffer-view-allocate-copy device '(2 3) :f32
                                                  (sb-sys:vector-sap a-bytes) (length a-bytes)))
                (bv-b (buffer-view-allocate-copy device '(3 2) :f32
                                                  (sb-sys:vector-sap b-bytes) (length b-bytes))))
            (unwind-protect
                 (with-call (call session "module.main")
                   (call-push-buffer-view call bv-a)
                   (call-push-buffer-view call bv-b)
                   (call-invoke call)
                   (let ((result (call-pop-buffer-view call)))
                     (unwind-protect
                          (is (eq :f32 (buffer-view-element-type result)))
                       (buffer-view-release result))))
              (buffer-view-release bv-a)
              (buffer-view-release bv-b))))))))

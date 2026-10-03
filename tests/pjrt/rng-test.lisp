;;;; rng-bit-generator の PJRT（XLA CPU プラグイン）での実行（issue #133、medium）。
;;;;
;;;; 実測の結果、PJRT の THREE_FRY は eager（= IREE の lowering を写した実装）と
;;;; ビット単位で一致した（docs/stablehlo-ops.md）。そのため eager と PJRT の
;;;; 一致をテストにする。状態の使い方（鍵 = [0]、カウンタ = [1]）も XLA が同じ。

(in-package #:nabla.pjrt.tests)

(defparameter *rng-pjrt-shapes*
  '(() (1) (2) (3) (7) (8) (1 1) (2 3) (3 2) (3 3) (4 3) (3 4) (5 1) (1 5)
    (2 3 5) (3 5 2) (3 3 3) (7 1 3) (2 2 2 2) (3 3 3 3) (16 16)))

(defun %rng-pjrt-state (seed)
  (let ((rs (sb-ext:seed-random-state seed))
        (state (make-array 2 :element-type '(unsigned-byte 64))))
    (setf (aref state 0) (random (expt 2 64) rs)
          (aref state 1) (if (zerop (mod seed 5))
                             (- (expt 2 64) 1 (random 20 rs))
                             (random (expt 2 64) rs)))
    state))

(define-pjrt-test backend/rng-bit-generator/matches-eager-bit-exactly
  "全ての形状・dtype について、PJRT の出力（新しい状態とビット）が eager と要素型も値も EQUALP で一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (%pjrt-backend)))
    (dolist (dtype '(:u32 :u64))
      (dolist (shape *rng-pjrt-shapes*)
        (let* ((graph (nb:trace-to-graph
                       (nb:with-tracing (s) (nb::rng-bit-generator s :shape shape :dtype dtype))
                       (list (nb:make-aval '(2) :u64))))
               (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
          (unwind-protect
               (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                             (lambda (seed)
                               (let ((state (%rng-pjrt-state seed)))
                                 (with-pjrt-arrays ((dstate (nabla:to-device state backend :dtype :u64)))
                                   (multiple-value-bind (new-state bits)
                                       (nabla:backend-invoke backend module "main" dstate)
                                     (unwind-protect
                                          (multiple-value-bind (eager-state eager-bits) (nb:eval-graph graph state)
                                            (and (equalp (nabla:to-host new-state) eager-state)
                                                 (equalp (nabla:to-host bits) eager-bits)
                                                 (equal (array-element-type (nabla:to-host bits))
                                                        (array-element-type eager-bits))))
                                       (release-device-array new-state)
                                       (release-device-array bits))))))
                             :regression-id backend/rng-bit-generator/matches-eager-bit-exactly
                             :regression-file (regression-path "pjrt-rng-bit-generator-matches-eager"
                                                               :package "NABLA.PJRT.TESTS"))
                   "shape ~S dtype ~S: PJRT の結果が eager と一致しなかった" shape dtype)
            (nabla:backend-unload backend module)))))))

;;;; PRNG の公開 API と、バッチ次元つき rng-bit-generator の PJRT（XLA CPU プラグイン）での実行
;;;; （issue #136、medium）。tests/iree/prng-test.lisp と同じ内容を PJRT で確かめる:
;;;; バッチ次元つきの状態（stablehlo.while で K 行ずつ rng_bit_generator を回す StableHLO）が
;;;; eager とビット単位で一致し、jit した uniform / normal / split / fold-in が eager と一致する。

(in-package #:nabla.pjrt.tests)

(defun %prng-pjrt-states (shape seed)
  (let ((rs (sb-ext:seed-random-state seed))
        (states (make-array shape :element-type '(unsigned-byte 64))))
    (dotimes (i (array-total-size states) states)
      (setf (row-major-aref states i) (random (expt 2 64) rs)))))

(define-pjrt-test backend/prng/batched-rng-bit-generator-matches-eager
  "バッチ次元つきの状態（行数 1・3・K-1・K+1・2K+3 と2段の (2 3)。K は while の1回で処理する行数）
× 形 × :u32 / :u64 で、PJRT の出力が eager とビット単位で一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (%pjrt-backend)))
    (dolist (state-shape (let ((k nb::*rng-rows-per-iteration*))
                           ;; K の倍数でない行数（while の最後の1回が重なる。issue #178）も含める
                           (list '(1 2) '(3 2) '(2 3 2) (list (max 1 (1- k)) 2) (list (1+ k) 2)
                                 (list (+ (* 2 k) 3) 2))))
      (dolist (shape '(() (3) (2 3)))
        (dolist (dtype '(:u32 :u64))
          (let* ((graph (nb:trace-to-graph
                         (nb:with-tracing (s) (nb::rng-bit-generator s :shape shape :dtype dtype))
                         (list (nb:make-aval state-shape :u64))))
                 (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph))))
                 (states (%prng-pjrt-states state-shape 5)))
            (unwind-protect
                 (with-pjrt-arrays ((dstate (nabla:to-device states backend :dtype :u64)))
                   (multiple-value-bind (new-state bits) (nabla:backend-invoke backend module "main" dstate)
                     (unwind-protect
                          (multiple-value-bind (eager-state eager-bits) (nb:eval-graph graph states)
                            (is (and (equalp (nabla:to-host new-state) eager-state)
                                     (equalp (nabla:to-host bits) eager-bits))
                                "状態 ~S・shape ~S・dtype ~S: PJRT が eager と一致しなかった"
                                state-shape shape dtype))
                       (release-device-array new-state)
                       (release-device-array bits))))
              (nabla:backend-unload backend module))))))))

(define-pjrt-test jit/prng-pjrt-matches-eager
  "PJRT で jit した split / fold-in はビット単位、uniform / normal は許容誤差つきで eager と一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (nabla:find-backend :pjrt))
        (key (nb:prng-key 2026)))
    (dolist (case (list (list "split" (nb:with-tracing (k) (nb:split k 5)) t)
                        (list "fold-in" (nb:with-tracing (k) (nb:fold-in k 9)) t)
                        (list "uniform" (nb:with-tracing (k) (nb:uniform k '(7 5))) nil)
                        (list "normal" (nb:with-tracing (k) (nb:normal k '(64 4))) nil)))
      (destructuring-bind (name fn exact) case
        (let* ((nb:*compile-cache-directory* nil)
               (jitted (nb:jit fn :backend backend))
               (compiled (funcall jitted key))
               (eager (funcall fn key)))
          (unwind-protect
               (if exact
                   (is (equalp compiled eager) "~A: PJRT の jit が eager と一致しなかった" name)
                   ;; exp / log の実装の差が erf の逆関数の裾で増幅されうるので rtol 1e-4
                   (is (allclose compiled eager :dtype :f32 :rtol 1d-4 :atol 1d-5)
                       "~A: PJRT の jit が eager と許容誤差内で一致しなかった" name))
            (nb::%jit-cache-forget (nb::%jitted-function-fn jitted))))))))

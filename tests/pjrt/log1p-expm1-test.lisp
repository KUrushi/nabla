;;;; log1p / expm1 を PJRT（XLA CPU）で jit しても小さな入力の桁を失わない（issue #223、medium）。
;;;;
;;;; f64 の emit は Goldberg / Kahan の式（primitives/unary.lisp）で、u を optimization_barrier に
;;;; 通さないと XLA の代数的な簡約が (x + 1) - 1 を x に畳み、素朴な log(1 + x) に戻る。

(in-package #:nabla.pjrt.tests)

(define-pjrt-test log1p-expm1/keep-the-digits-of-small-inputs-under-jit
  "PJRT で jit した log1p / expm1 は、小さな |x| でも eager（libm）と相対誤差 8 ulp で一致する（f32 と f64）。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (%pjrt-backend))
        (nabla:*compile-cache-directory* nil))
    (dolist (dtype '(:f32 :f64))
      (let* ((type (if (eq dtype :f64) 'double-float 'single-float))
             (values (if (eq dtype :f64)
                         '(4d-18 -3d-17 1d-12 -1d-10 1d-300 0.5d0 -0.75d0 3d0 -0d0)
                         '(1d-10 -3d-9 1d-6 -1d-5 1d-30 0.5d0 -0.75d0 3d0 -0d0)))
             (x (make-array (length values) :element-type type
                                            :initial-contents (mapcar (lambda (v) (coerce v type)) values)))
             (rtol (if (eq dtype :f64) 1d-15 1d-6)))
        (dolist (entry (list (cons "log1p" (nabla:with-tracing (x) (nabla:log1p x)))
                             (cons "expm1" (nabla:with-tracing (x) (nabla:expm1 x)))))
          (let ((jitted (funcall (nabla:jit (cdr entry) :backend backend) x))
                (eager (funcall (cdr entry) x)))
            (is (every (lambda (a e)
                         (if (zerop e)
                             (and (zerop a) (= (float-sign a) (float-sign e)))
                             (<= (abs (- a e)) (* rtol (abs e)))))
                       jitted eager)
                "~A/~A: jit ~S, eager ~S" (car entry) dtype jitted eager)))))))

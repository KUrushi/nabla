;;;; scan を jvp 変換した graph を IREE（local）でコンパイル・実行し、eager
;;;; （eval-graph）と比べる medium テスト（issue #135）。tests/iree/scan-test.lisp の
;;;; %WITH-SCAN-IREE-CHECK を使う。

(in-package #:nabla.iree.tests)

(defun %jvp-scan-iree-graph (n length reverse)
  "carry = (h, g, c:i32)、consts = (w)、xs = (u, x2) の f32 の scan（g は h を受けるので、
g の接線は初期ゼロでも非ゼロになる）。"
  (nb::trace-to-graph
   (nb:with-tracing (w h g c u x2)
     (multiple-value-bind (carry ys)
         (nb:scan (nb:with-tracing (carry x)
                    (let ((h (first carry)) (g (second carry)) (c (third carry))
                          (u (first x)) (x2 (second x)))
                      (values (list (tanh (+ (* h w) u)) (+ (* g 0.9) h) (+ c 1))
                              (list (* h u) (* x2 2.0)))))
                  (list h g c) (list u x2) :length length :reverse reverse)
       (values (first carry) (second carry) (third carry) (first ys) (second ys))))
   (list (nb:make-aval (list n) :f32) (nb:make-aval (list n) :f32) (nb:make-aval (list n) :f32)
         (nb:make-aval '() :i32)
         (nb:make-aval (list length n) :f32) (nb:make-aval (list length n) :f32))))

(define-iree-test jvp-scan/iree-all-tangents-match-eager
    "全ての浮動小数点の入力に接線がある scan の jvp graph（forward と reverse）が、IREE の実行結果と eager で一致する。"
  (skip-unless-iree :library :both)
  (dolist (reverse '(nil t))
    (let ((graph (nb::jvp-graph (%jvp-scan-iree-graph 3 4 reverse))))
      (%with-scan-iree-check (graph "jvp-all") "IREE の scan の jvp graph の結果が eager と一致しなかった"))))

(define-iree-test jvp-scan/iree-partial-tangents-match-eager
    "xs の u にだけ接線がある（g と h の carry の接線は不動点で非ゼロになる）scan の jvp graph も、
IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::jvp-graph (%jvp-scan-iree-graph 3 4 t) :nonzero '(nil nil nil nil t nil))))
    (%with-scan-iree-check (graph "jvp-partial") "IREE の部分的な接線の scan の jvp graph の結果が eager と一致しなかった")))

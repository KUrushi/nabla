;;;; with-tracing の do ループ（scan に展開される）を IREE の local backend で
;;;; コンパイル・実行し、eager（eval-graph）と比べる medium テスト（issue #137）。
;;;; %with-scan-iree-check は tests/iree/scan-test.lisp のもの。

(in-package #:nabla.iree.tests)

(defun %loop-scan-iree-graph (n)
  "h <- tanh(h*s + 0.1) と k <- k+1 を n 回回す do ループを含む graph。入力は (h0 w)。"
  (nb::trace-to-graph
   (nb:with-tracing (h0 w)
     (do ((i 0 (1+ i))
          (h h0 (tanh (+ (* h s) 0.1)))
          (s w)
          (k 0.0 (+ k 1.0)))
         ((>= i n) (values h k))))
   (list (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32))))

(define-iree-test loop-scan/iree-do-loop-matches-eager
    "do ループが展開された scan の StableHLO を IREE でコンパイル・実行した結果が eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (%loop-scan-iree-graph 5)))
    (%with-scan-iree-check (graph "do-loop") "IREE の do ループの結果が eager と一致しなかった")))

(define-iree-test loop-scan/iree-do-loop-zero-iterations-matches-eager
    "反復回数 0 の do ループも IREE と eager で一致する（carry は素通し）。"
  (skip-unless-iree :library :both)
  (let ((graph (%loop-scan-iree-graph 0)))
    (%with-scan-iree-check (graph "do-loop-zero") "IREE の反復0回の do ループの結果が eager と一致しなかった")))

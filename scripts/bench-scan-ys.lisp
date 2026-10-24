;;;; 本体に制御構造がある scan の、1回の呼び出し（backend-invoke）の時間を、外側の長さ L と
;;;; ys の大きさを変えて測る（issue #233）。
;;;;
;;;; #220 は、本体に制御構造（while / if）がある scan には ys の barrier を付けない（IREE 3.11 の
;;;; local-task の競合の回避。docs/iree-upstream-bugs.md の 7）。代償は、ys が更新のたびに丸ごと
;;;; コピーされること（長さの2乗）。その大きさを測り、改善の候補の emit を比べるためのもの。
;;;; scripts/bench-backends.lisp を先に読み、その統計・レコード・表の関数を使う。
;;;; scripts/bench-scan-ys.sh 経由で使う（直接なら profile-call-overhead.lisp と同じ流儀）:
;;;;
;;;;   sbcl --non-interactive --eval '(require :asdf)' --load scripts/bench-backends.lisp \
;;;;     --load scripts/bench-scan-ys.lisp --eval '(nabla-bench::scan-ys-main)' > scan-ys.sexp
;;;;
;;;; 測るのは「main(h: f32[W]) -> (carry f32[W], ys f32[L, W])」の scan を L 回回す関数
;;;; （xs は無い。ys の大きさ = L * W * 4 バイト）。variant（列）は:
;;;;   plain   本体に制御構造の無い scan（tanh）。ys の barrier が付く（いちばん速い基準）
;;;;   cond    本体に cond* がある scan。ys の barrier が付かない（今の既定の emit。#220 の回避策）
;;;;   <path>  StableHLO のテンプレートのファイル。@L@ と @W@ を数で置き換えて、同じ形の main として測る
;;;;           （改善の候補の emit を、nabla の emit を変えずに比べる。scripts/scan-ys-candidates/）
;;;; 環境変数:
;;;;   NABLA_BENCH_SCAN_LENGTHS   L のリスト（既定 16,32,64,128）
;;;;   NABLA_BENCH_SCAN_WIDTHS    W のリスト（既定 4096,65536。L=64 のとき ys が 1 MB と 16 MB）
;;;;   NABLA_BENCH_SCAN_VARIANTS  variant のリスト（既定 plain,cond）
;;;;   NABLA_BENCH_SCAN_REPS      各セルの測定回数（既定 7。最初の1回は捨てる）

(in-package #:nabla-bench)

(asdf:load-system "nabla/iree")          ; NB: のシンボルを読むので、先に読み込む

(defun scan-ys-graph (kind length width)
  "KIND（:plain か :cond）の本体を LENGTH 回回す scan の graph。h は f32[WIDTH]、ys に各ステップの h を積む。"
  (nb::trace-to-graph
   (nb:with-tracing (h)
     (multiple-value-bind (carry ys)
         (nb:scan (nb:with-tracing (carry x)
                    x
                    (let ((h (if (eq kind :cond)
                                 (nb:cond* (< (nb:reduce-sum (first carry)) 0.5)
                                           (nb:with-tracing (c) (tanh c))
                                           (nb:with-tracing (c) (tanh (tanh c)))
                                           (first carry))
                                 (tanh (first carry)))))
                      (values (list h) (list h))))
                  (list h) '() :length length)
       (values (first carry) (first ys))))
   (list (nb:make-aval (list width) :f32))))

(defun read-file-string (path)
  (with-open-file (in path)
    (let ((s (make-string (file-length in))))
      (subseq s 0 (read-sequence s in)))))

(defun replace-all (string old new)
  (with-output-to-string (out)
    (loop with start = 0
          for hit = (search old string :start2 start)
          do (write-string string out :start start :end (or hit (length string)))
          while hit do (write-string new out) (setf start (+ hit (length old))))))

(defun scan-ys-text (variant length width)
  "VARIANT の StableHLO テキスト。"
  (cond ((member variant '("plain" "cond") :test #'string=)
         (nabla-call "EMIT-STABLEHLO"
                     (scan-ys-graph (if (string= variant "plain") :plain :cond) length width)))
        (t (replace-all (replace-all (read-file-string variant) "@L@" (princ-to-string length))
                        "@W@" (princ-to-string width)))))

(defun scan-ys-main ()
  (let* ((lengths (mapcar #'parse-integer (parse-list-env "NABLA_BENCH_SCAN_LENGTHS" '("16" "32" "64" "128"))))
         (widths (mapcar #'parse-integer (parse-list-env "NABLA_BENCH_SCAN_WIDTHS" '("4096" "65536"))))
         (variants (parse-list-env "NABLA_BENCH_SCAN_VARIANTS" '("plain" "cond")))
         (reps (parse-integer (or (env-var "NABLA_BENCH_SCAN_REPS") "7"))))
    (handler-case
        (progn
          (setf (symbol-value (find-symbol "*COMPILE-CACHE-DIRECTORY*" "NABLA")) nil)
          (let ((backend (make-backend-for "iree-local")))
            (funcall (sym "NABLA.IREE" "RELEASE-DEVICE-ARRAY")
                     (nabla-call "TO-DEVICE" (make-array 1 :element-type 'single-float) backend))
            (emit-env "iree-local" :iree)
            (dolist (width widths)
              (dolist (length lengths)
                (let ((config (format nil "L~D-W~D (ys ~,2F MB)" length width (/ (* length width 4) 1048576.0)))
                      (input (nabla-call "TO-DEVICE" (make-array width :element-type 'single-float :initial-element 0.3) backend)))
                  (dolist (variant variants)
                    (let ((label (if (member variant '("plain" "cond") :test #'string=)
                                     variant (pathname-name (pathname variant)))))
                      (handler-case
                          (let* ((module (nabla-call "BACKEND-LOAD" backend
                                                     (nabla-call "BACKEND-COMPILE" backend (scan-ys-text variant length width))))
                                 (samples '()))
                            (dotimes (i (1+ reps))
                              (let ((ms (elapsed-ms (mapc (sym "NABLA.IREE" "RELEASE-DEVICE-ARRAY")
                                                          (multiple-value-list
                                                           (nabla-call "BACKEND-INVOKE" backend module "main" input))))))
                                (when (plusp i) (push ms samples))))
                            (nabla-call "BACKEND-UNLOAD" backend module)
                            (emit (result-record label config "invoke" (reverse samples)))
                            (emit (result-record label config "invoke/per-step"
                                                 (mapcar (lambda (ms) (/ ms length)) (reverse samples)))))
                        (error (c)
                          (format *error-output* "~&[scan-ys] ~A ~A: ~A~%" label config (one-line (princ-to-string c)))
                          (emit (unmeasured-record label (one-line (format nil "error: ~A" c))))))))
                  (funcall (sym "NABLA.IREE" "RELEASE-DEVICE-ARRAY") input))))))
      (error (c)
        (emit (unmeasured-record "iree-local" (one-line (format nil "error: ~A" c))))))))

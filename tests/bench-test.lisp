;;;; scripts/bench-backends.lisp の出力の読み書きと表の組み立て（issue #89）。
;;;; 数値そのものは主張しない。固定の文字列を与えて、項目が揃うことだけを確かめる。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defvar *bench-loaded* nil)

(defun %bench-fn (name)
  "scripts/bench-backends.lisp を（初回のみ）load して、パッケージ NABLA-BENCH の関数 NAME を返す。"
  (unless *bench-loaded*
    (load (asdf:system-relative-pathname "nabla" "scripts/bench-backends.lisp"))
    (setf *bench-loaded* t))
  (symbol-function (find-symbol name "NABLA-BENCH")))

(defparameter *bench-sample*
  "(:kind :env :backend \"pjrt-cpu\" :cpu \"Test CPU\" :cores 4)
; コメント行は読み飛ばす

(:kind :result :backend \"iree-local\" :config \"small\" :metric \"step/full\" :unit \"ms\" :n 200 :median 0.5d0 :p10 0.4d0 :p90 0.7d0 :min 0.3d0)
(:kind :result :backend \"pjrt-cpu\" :config \"small\" :metric \"step/full\" :unit \"ms\" :n 200 :median 0.25d0 :p10 0.2d0 :p90 0.3d0 :min 0.1d0)
(:kind :result :backend \"iree-local\" :config \"small\" :metric \"stage/backend-compile\" :unit \"ms\" :n 1 :median 1234.5d0 :p10 1234.5d0 :p90 1234.5d0 :min 1234.5d0)
(:kind :result :backend \"iree-cuda\" :config \"-\" :metric \"-\" :status :unmeasured :reason \"no GPU\")
")

(test bench/parse-records-reads-env-and-results
  "固定のレコード列から、env 1件と result 4件（コメント・空行は無視）が plist として読める。"
  (let ((records (funcall (%bench-fn "PARSE-RECORDS") *bench-sample*)))
    (is (= 5 (length records)))
    (is (= 1 (count :env records :key (lambda (r) (getf r :kind)))))
    (is (equal "Test CPU" (getf (first records) :cpu)))
    (let ((record (second records)))
      (is (equal '("iree-local" "small" "step/full" 200)
                 (list (getf record :backend) (getf record :config) (getf record :metric) (getf record :n))))
      (is (= 0.5d0 (getf record :median))))
    (is (eq :unmeasured (getf (fifth records) :status)))))

(test bench/parse-records-rejects-non-records
  "READ-EVAL は無効で、:kind の無い行は ERROR になる。"
  (signals error (funcall (%bench-fn "PARSE-RECORDS") "(:backend \"x\")"))
  (signals error (funcall (%bench-fn "PARSE-RECORDS") "#.(+ 1 2)")))

(test bench/summarize-percentiles
  "1..11 の中央値は 6、p10 は 2、p90 は 10、最小は 1（線形補間）。"
  (let ((summary (funcall (%bench-fn "SUMMARIZE") (loop for i from 1 to 11 collect i))))
    (is (equal '(11 6 2 10 1) (list (getf summary :n) (getf summary :median)
                                    (getf summary :p10) (getf summary :p90) (getf summary :min))))))

(test bench/records-table-has-all-backends-and-metrics
  "表には全 backend の列と全 metric の行があり、未測定の backend は「未測定」と出る。"
  (let ((table (funcall (%bench-fn "RECORDS-TABLE") (funcall (%bench-fn "PARSE-RECORDS") *bench-sample*))))
    (dolist (needle '("### small" "iree-local" "pjrt-cpu" "iree-cuda" "step/full" "stage/backend-compile"
                      "0.500 [0.400-0.700]" "0.250" "1234.500" "未測定"))
      (is (search needle table) "表に ~S が無い:~%~A" needle table))))

(test bench/result-record-roundtrips-through-parse
  "RESULT-RECORD の出力を PRINT-RECORD で書いて PARSE-RECORDS で読み戻すと、項目が同じ。"
  (let* ((record (funcall (%bench-fn "RESULT-RECORD") "b" "c" "m" (list 1d0 2d0 3d0)))
         (text (with-output-to-string (s) (funcall (%bench-fn "PRINT-RECORD") record s)))
         (back (first (funcall (%bench-fn "PARSE-RECORDS") text))))
    (is (equal record back))
    (is (= 2d0 (getf back :median)))))

(test bench/unmeasured-reason-with-newlines-stays-one-record
  "複数行のエラー文（コンパイルエラーなど）を理由に入れても、1行1レコードのまま読み戻せる。"
  (let* ((reason (funcall (%bench-fn "ONE-LINE") (format nil "error: a~%  b~%~%c")))
         (text (with-output-to-string (s)
                 (funcall (%bench-fn "PRINT-RECORD") (funcall (%bench-fn "UNMEASURED-RECORD") "b" reason) s)))
         (records (funcall (%bench-fn "PARSE-RECORDS") text)))
    (is (equal "error: a b c" reason))
    (is (= 1 (length records)))))

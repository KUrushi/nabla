;;;; CPU プラグインの CFFI 疎通（例ベース。issue #78）。
;;;;
;;;; dlopen して GetPjrtApi を呼び、PJRT_Api 先頭の pjrt_api_version を読む。
;;;; 固定した xla-cpu-pjrt 0.0.1 は PJRT API 0.81。

(in-package #:nabla.pjrt.tests)

(define-pjrt-test cpu-plugin-api-version
  "CPU プラグインの API の版は major 0、minor は 81 以上。"
  (skip-unless-pjrt :kind :cpu)
  (let ((api (load-plugin :cpu)))
    (is (not (cffi:null-pointer-p api)))
    (multiple-value-bind (major minor) (plugin-api-version api)
      (is (= 0 major))
      (is (>= minor 81) "minor = ~D" minor))
    ;; 二度ロードしても同じ API を得る
    (is (cffi:pointer-eq api (load-plugin :cpu)))))

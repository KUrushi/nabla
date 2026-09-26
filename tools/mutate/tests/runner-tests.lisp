;;;; runner-tests.lisp -- run / exclusions の受け入れテスト
;;;;
;;;; tools/mutate/sample の弱いテスト・強いテストに対して runner を
;;;; 実際に走らせ、弱いテストでは生き残る変異体があり、強いテストでは
;;;; 全滅することを確かめる。ファイルはサンプルシステムを ASDF 経由で
;;;; 探さず、tools/mutate 自身の場所からの相対パスで直接 `load` する
;;;; （CL_SOURCE_REGISTRY の設定に左右されないようにするため）。

(in-package #:nabla.mutate.tests)

(in-suite :nabla-mutate)

(defun %sample-path (relative)
  (merge-pathnames relative (asdf:system-source-directory "nabla-mutate")))

(defun %load-sample ()
  (load (%sample-path "sample/src/sample.lisp"))
  (load (%sample-path "sample/weak-tests/weak.lisp"))
  (load (%sample-path "sample/strong-tests/strong.lisp")))

(test excluded-p-matches-by-file-suffix-and-mutation-string
  (let ((entry (list :file "src/sample.lisp" :form nil
                      :mutation "5 -> 0" :reason "テスト用")))
    (is-true (nabla.mutate:excluded-p (list entry) #P"/tmp/foo/src/sample.lisp" :constant 5 0))
    (is-false (nabla.mutate:excluded-p (list entry) #P"/tmp/foo/src/sample.lisp" :constant 5 1))
    (is-false (nabla.mutate:excluded-p (list entry) #P"/tmp/foo/src/other.lisp" :constant 5 0))))

(test default-exclusions-file-loads-as-a-list-of-plists
  (let ((exclusions (nabla.mutate:load-exclusions (nabla.mutate:default-exclusions-path))))
    (is (listp exclusions))
    (dolist (entry exclusions)
      (is (stringp (getf entry :file)))
      (is (stringp (getf entry :mutation)))
      (is (stringp (getf entry :reason))))))

(test run-reports-survivors-with-weak-tests-and-none-with-strong-tests
  (%load-sample)
  (let* ((sample-file (%sample-path "sample/src/sample.lisp"))
         (weak-report (nabla.mutate:run
                       :files (list sample-file)
                       :test-function (lambda ()
                                        (funcall (find-symbol "RUN-TESTS" "NABLA.MUTATE.SAMPLE.WEAK-TESTS")))
                       :exclusions nil
                       :timeout-seconds 10
                       :trials 5
                       :stream (make-broadcast-stream)))
         (strong-report (nabla.mutate:run
                         :files (list sample-file)
                         :test-function (lambda ()
                                          (funcall (find-symbol "RUN-TESTS" "NABLA.MUTATE.SAMPLE.STRONG-TESTS")))
                         :exclusions (nabla.mutate:default-exclusions-path)
                         :timeout-seconds 10
                         :trials 5
                         :stream (make-broadcast-stream))))
    (is (> (length (nabla.mutate:report-mutants weak-report)) 0)
        "サンプルには変異可能な定義があるはず")
    (is (> (count :survived (nabla.mutate:report-mutants weak-report)
                  :key #'nabla.mutate:mutant-status)
           0)
        "弱いテストでは生き残る変異体があるはず")
    (is (= 0 (count :survived (nabla.mutate:report-mutants strong-report)
                     :key #'nabla.mutate:mutant-status))
        "強いテストでは、除外リストにある等価変異体を除いてすべて殺されるはず")
    (is (> (count :excluded (nabla.mutate:report-mutants strong-report)
                   :key #'nabla.mutate:mutant-status)
           0)
        "clamp の下限チェックの等価変異体は除外リストで除外されるはず")))

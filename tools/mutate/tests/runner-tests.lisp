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

(test evaluate-mutant-treats-control-stack-exhaustion-as-killed
  "暴走再帰で SB-KERNEL::CONTROL-STACK-EXHAUSTED（ERROR ではなく
STORAGE-CONDITION）を投げる変異体は、RUN 全体を中断させず :killed として
判定される。arith-swap が `(- n 1)` を `(+ n 1)` にしたときのような
実例に対応する。"
  (let* ((package (find-package "NABLA.MUTATE.TESTS"))
         (name (intern "%RUNNER-TESTS-COUNTDOWN" package))
         (original `(defun ,name (n) (if (<= n 0) 0 (1+ (,name (- n 1))))))
         (mutated `(defun ,name (n) (if (<= n 0) 0 (1+ (,name (+ n 1))))))
         (test-function (lambda () (= 0 (funcall name 5)))))
    (unwind-protect
         (is (eq :killed
                 (nabla.mutate::%evaluate-mutant original mutated package test-function 10)))
      (ignore-errors (fmakunbound name))
      (ignore-errors (unintern name package)))))

(test evaluate-mutant-does-not-leave-extra-method-after-eql-specializer-mutation
  "(defmethod g ((x (eql 0))) ...) を変異させて評価しても、%evaluate-mutant
から戻ったあとには元の1メソッドしか残らない。specializer を持つ
lambda list は arid として変異させないので、mutate-form 自体が
適用できず（applied=NIL）、original-form をそのまま評価しても
新しいメソッドは増えない。"
  (let* ((package (find-package "NABLA.MUTATE.TESTS"))
         (name (intern "%RUNNER-TESTS-EQL-SPECIALIZER-TARGET" package))
         (original `(defmethod ,name ((x (eql 0))) :zero)))
    (eval `(defgeneric ,name (x)))
    (eval original)
    (unwind-protect
         (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form original :constant)
           (is-false applied
                      "specializer の中の 0 は arid なので変異が適用できないはず")
           (is (equal original mutated))
           ;; それでも %evaluate-mutant を経由した1サイクル（元の定義を
           ;; 再評価するだけ）でメソッドが増えないことを確かめる。
           (nabla.mutate::%evaluate-mutant original mutated package (lambda () t) 5)
           (is (= 1 (length (sb-mop:generic-function-methods (fdefinition name)))))
           (is (eq :zero (funcall name 0))))
      (ignore-errors (fmakunbound name))
      (ignore-errors (unintern name package)))))

(test isolate-mutant-side-effects-leaves-no-file-and-no-plist-residue
  "check-it の :regression-file / :regression-id を使うテストが1つの
変異体の評価中に落ちても、ファイルにも check-it::regression-cases
plist にも痕跡が残らない（%evaluate-mutant から戻ったあとに元へ
戻される）。"
  (let* ((dir (merge-pathnames "tmp-regression-isolation-test/"
                                (asdf:system-source-directory "nabla-mutate")))
         (file (merge-pathnames "case.lisp" dir))
         (rid (intern "%RUNNER-TESTS-ISOLATION-REGRESSION-PROP" "NABLA.MUTATE.TESTS"))
         (package (find-package "NABLA.MUTATE.TESTS"))
         (original '(defun %runner-tests-isolation-noop-target () 1))
         (mutated '(defun %runner-tests-isolation-noop-target () 2))
         (failing-test (lambda ()
                          (check-it (generator (integer 0 0))
                                    (lambda (n) (declare (ignore n)) nil)
                                    :regression-id %runner-tests-isolation-regression-prop
                                    :regression-file file))))
    (ignore-errors (uiop:delete-directory-tree dir :validate t))
    (ensure-directories-exist dir)
    (with-open-file (s file :direction :output :if-does-not-exist :create)
      (format s "~&(in-package #:nabla.mutate.tests)~%"))
    (unwind-protect
         (let ((file-before (alexandria:read-file-into-string file))
               (plist-before (get rid 'check-it::regression-cases))
               (nabla.mutate:*regression-directories* (list dir)))
           (is (eq :killed
                   (nabla.mutate::%evaluate-mutant original mutated package failing-test 5)))
           (is (equal file-before (alexandria:read-file-into-string file))
               "regression ファイルに何も書き込まれずに残っているはず")
           (is (equal plist-before (get rid 'check-it::regression-cases))
               "check-it::regression-cases plist が元の状態に戻っているはず"))
      (ignore-errors (remprop rid 'check-it::regression-cases))
      (ignore-errors (uiop:delete-directory-tree dir :validate t)))))

(test isolate-mutant-side-effects-makes-result-order-independent
  "ある変異体（M1）の評価で check-it が regression-case を記録しても、
その後に評価する無関係な別の変異体（M2）の判定はそれに左右されない
（M2 を単体で走らせたときと同じ結果になる）。M1 が積んだ regression-case
が M2 の check-it::regression-cases に漏れて再生され、M2 だけを走らせれば
決して失敗しない性質を偽って falsely kill するのが直したバグ。"
  (let* ((dir (merge-pathnames "tmp-regression-order-test/"
                                (asdf:system-source-directory "nabla-mutate")))
         (file1 (merge-pathnames "m1.lisp" dir))
         (file2 (merge-pathnames "m2.lisp" dir))
         (rid (intern "%RUNNER-TESTS-ORDER-REGRESSION-PROP" "NABLA.MUTATE.TESTS"))
         (package (find-package "NABLA.MUTATE.TESTS"))
         (original '(defun %runner-tests-order-noop-target () 1))
         (mutated '(defun %runner-tests-order-noop-target () 2))
         (m1-test (lambda ()
                     ;; 0 を生成域に持ち、必ず失敗して datum "0" を
                     ;; regression として記録する。
                     (check-it (generator (integer 0 0))
                               (lambda (n) (declare (ignore n)) nil)
                               :regression-id %runner-tests-order-regression-prop
                               :regression-file file1)))
         (m2-test (lambda ()
                     ;; 0 を生成しない限り必ず通るが、M1 が漏らした
                     ;; datum 0 が再生されると失敗する。
                     (check-it (generator (integer 1 5))
                               (lambda (n) (/= n 0))
                               :regression-id %runner-tests-order-regression-prop
                               :regression-file file2))))
    (ignore-errors (uiop:delete-directory-tree dir :validate t))
    (ensure-directories-exist dir)
    (dolist (file (list file1 file2))
      (with-open-file (s file :direction :output :if-does-not-exist :create)
        (format s "~&(in-package #:nabla.mutate.tests)~%")))
    (unwind-protect
         (let ((nabla.mutate:*regression-directories* (list dir)))
           (is (eq :killed
                   (nabla.mutate::%evaluate-mutant original mutated package m1-test 5))
               "M1 は毎回失敗する性質なので killed のはず")
           (is (eq :survived
                   (nabla.mutate::%evaluate-mutant original mutated package m2-test 5))
               "M1 の結果に関係なく、M2 は単体で走らせたのと同じ survived になるはず"))
      (ignore-errors (remprop rid 'check-it::regression-cases))
      (ignore-errors (uiop:delete-directory-tree dir :validate t)))))

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

(test plan-mutants-is-deterministic-distinct-and-finer-than-definitions
  "PLAN-MUTANTS（テストを走らせない数え上げ）は、同じ入力に同じ順の
変異体を返し、1つの定義の中で変異後のフォームが重複せず、サンプルの
定義の数（2）よりはっきり多くの変異体を作る。"
  (%load-sample)
  (let* ((files (list (%sample-path "sample/src/sample.lisp")))
         (plan1 (nabla.mutate:plan-mutants :files files))
         (plan2 (nabla.mutate:plan-mutants :files files)))
    (is (equal (mapcar #'nabla.mutate:mutant-mutated-form plan1)
               (mapcar #'nabla.mutate:mutant-mutated-form plan2)))
    (is (> (length plan1) (* 2 2)))
    (dolist (line (remove-duplicates (mapcar #'nabla.mutate:mutant-line plan1)))
      (let ((forms (mapcar #'nabla.mutate:mutant-mutated-form
                           (remove line plan1 :key #'nabla.mutate:mutant-line :test #'/=))))
        (is (%distinct-p forms))))))

(test plan-mutants-respects-per-definition-cap
  (is (check-it
       (generator (integer 1 4))
       (lambda (cap)
         (let ((plan (nabla.mutate:plan-mutants
                      :files (list (%sample-path "sample/src/sample.lisp"))
                      :max-mutants-per-definition cap)))
           (every (lambda (line)
                    (<= (count line plan :key #'nabla.mutate:mutant-line) cap))
                  (mapcar #'nabla.mutate:mutant-line plan)))))))

(test plan-mutants-covers-every-definition-in-a-file-with-several-batch-rules
  "def-batch-rule と defun が混ざった1つのファイルで、変異体を作れる
定義がすべて（最初の1つだけでなく）変異される。定義ごとに行番号で数える。"
  (let ((tmp (uiop:with-temporary-file (:pathname p :type "lisp" :keep t) p)))
    (unwind-protect
         (progn
           (with-open-file (s tmp :direction :output :if-exists :supersede)
             (write-string
              (format nil "(in-package #:nabla.mutate.tests)~%~
(def-batch-rule a (args batch-dims) (values (list (+ 1 2)) batch-dims))~%~
(defun helper (x) (+ x 1))~%~
(def-batch-rule b (args batch-dims &key k) (values (list (- k 1)) batch-dims))~%~
(def-batch-rule c (args batch-dims) (values args (list (* 2 3))))~%")
              s))
           (let ((plan (nabla.mutate:plan-mutants :files (list tmp))))
             (is (equal '(2 3 4 5)
                        (sort (remove-duplicates (mapcar #'nabla.mutate:mutant-line plan)) #'<)))))
      (ignore-errors (delete-file tmp)))))

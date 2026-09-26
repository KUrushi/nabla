;;;; runner.lisp -- 変異体を作り、テストスイートを回して判定する

(in-package #:nabla.mutate)

(defstruct mutant
  "1つの変異体の記録。STATUS は :killed / :survived / :excluded / :timeout。"
  file line original-form mutated-form operator status)

(setf (documentation 'mutant-file 'function) "その変異体があるファイルのパス。")
(setf (documentation 'mutant-line 'function) "その変異体の定義が始まる行番号（1始まり）。")
(setf (documentation 'mutant-original-form 'function) "変異させる前の、トップレベルの定義。")
(setf (documentation 'mutant-mutated-form 'function) "変異させた後の、トップレベルの定義。")
(setf (documentation 'mutant-operator 'function) "適用した変異演算子（*MUTATION-OPERATORS* のいずれか）。")
(setf (documentation 'mutant-status 'function)
      "判定結果。:killed / :survived / :excluded / :timeout のいずれか。")

(defstruct report
  "RUN の結果。MUTANTS は MUTANT のリスト。"
  mutants)

(setf (documentation 'report-mutants 'function) "この RUN で作られた MUTANT のリスト。")

(defun mutation-score (report)
  "mutation score を有理数で返す。殺した数（timeout を含む） ÷
（全数 − 除外数）。殺せる変異体が1つもない（全数=除外数）ときは 1 とする。"
  (let* ((mutants (report-mutants report))
         (total (length mutants))
         (excluded (count :excluded mutants :key #'mutant-status))
         (killed (count-if (lambda (m) (member (mutant-status m) '(:killed :timeout)))
                            mutants)))
    (if (= total excluded)
        1
        (/ killed (- total excluded)))))

(defun default-test-function ()
  "既定のテスト実行関数。`nabla.tests.support:run-tests` を実行時に
find-package / find-symbol で探して呼ぶ。#4（テスト基盤）が
マージされる前は、この関数を呼ばず :test-function を明示的に渡すこと。"
  (let ((package (find-package "NABLA.TESTS.SUPPORT")))
    (unless package
      (error "package NABLA.TESTS.SUPPORT が見つからない。~
:test-function に実行したいテスト関数を渡すこと。"))
    (let ((run-tests (find-symbol "RUN-TESTS" package)))
      (unless run-tests
        (error "NABLA.TESTS.SUPPORT:RUN-TESTS が見つからない"))
      (funcall run-tests :sizes '(:small :medium)))))

(defun %normalize-exclusions (exclusions)
  (if (or (pathnamep exclusions) (stringp exclusions))
      (load-exclusions exclusions)
      exclusions))

(defun %ranges-for-run (files ranges base-ref)
  (cond
    (ranges ranges)
    (files (mapcar (lambda (f) (list f 1 most-positive-fixnum)) files))
    (t (ranges-from-git-diff :base-ref base-ref))))

(defun %collect-candidates (range-list)
  "RANGE-LIST（(path start end) のリスト）から、対象になる定義を
(path . source-form) のリストとして重複なく返す。"
  (let ((forms-by-file (make-hash-table :test #'equal))
        (seen (make-hash-table :test #'equal))
        (candidates nil))
    (dolist (range range-list)
      (destructuring-bind (path start end) range
        (let* ((key (namestring path))
               (forms (or (gethash key forms-by-file)
                          (setf (gethash key forms-by-file) (read-source-forms path)))))
          (dolist (def (definitions-in-range forms start end))
            (let ((seen-key (list key (source-form-start-line def))))
              (unless (gethash seen-key seen)
                (setf (gethash seen-key seen) t)
                (push (cons path def) candidates)))))))
    (nreverse candidates)))

(defun %eval-quietly (form)
  "FORM を評価する。同じ名前の再定義など、無害な warning は黙らせる
（変異体ごとに元の定義へ戻すため、同じ defun を何度も再評価する）。"
  (handler-bind ((warning #'muffle-warning))
    (eval form)))

(defun %evaluate-mutant (original-form mutated-form package test-function timeout-seconds)
  "MUTATED-FORM を PACKAGE の中で評価し、TEST-FUNCTION を走らせて
:killed / :survived / :timeout を返す。呼び終わったら（成功しても
失敗しても）ORIGINAL-FORM を評価し直して元に戻す。"
  ;; ERROR だけでなく STORAGE-CONDITION（SB-KERNEL::CONTROL-STACK-EXHAUSTED
  ;; など）も捕まえる。どちらも SERIOUS-CONDITION のサブタイプだが、
  ;; STORAGE-CONDITION は ERROR のサブタイプではないため、`(error () ...)`
  ;; だけでは暴走再帰を作る変異体（例: `(- n 1)` → `(+ n 1)`）で
  ;; RUN 全体が中断してしまう。
  (let ((*package* package))
    (unwind-protect
         (handler-case
             (progn
               (%eval-quietly mutated-form)
               (handler-case
                   (if (sb-ext:with-timeout timeout-seconds (funcall test-function))
                       :survived
                       :killed)
                 (sb-ext:timeout () :timeout)
                 (serious-condition () :killed)))
           (serious-condition () :killed))
      (ignore-errors (%eval-quietly original-form)))))

(defun %check-it-trials-symbol ()
  (let ((package (find-package "CHECK-IT")))
    (and package (find-symbol "*NUM-TRIALS*" package))))

(defun %status-for (original mutated package test-function timeout-seconds trials)
  "変異体1体の最終状態を決める。まず TRIALS 回に減らして走らせ、
生き残ったら（テストが落ちなかったら）既定の試行回数で再確認する。"
  (let ((trial-sym (%check-it-trials-symbol))
        (evaluate (lambda ()
                    (%evaluate-mutant original mutated package test-function timeout-seconds))))
    (let ((first-status (if trial-sym
                             (progv (list trial-sym) (list trials) (funcall evaluate))
                             (funcall evaluate))))
      (if (eq first-status :survived)
          (funcall evaluate)
          first-status))))

(defun %check-baseline (test-function)
  "変異をかける前に、TEST-FUNCTION が（変異なしの状態で）エラーなく
走り、かつ通ることを確かめる。ここでエラーになったり落ちたりするのは、
たいてい TEST-FUNCTION の設定ミス（対象システムを読み込んでいない、
DEFAULT-TEST-FUNCTION が探すパッケージが存在しないなど）であって、
コードの問題ではない。ここで気づかず変異体の判定に混ぜると、
すべての変異体が黙って :killed になり、mutation score が見かけ上
1 になってしまう（.claude/skills/nabla-testing/references/mutation.md
の「1. 手順」）。"
  (let ((result (handler-case (funcall test-function)
                  (error (e)
                    (error "mutation testing を始める前に、既定のテストスイートが~
エラーで終わった。:test-function か :test-system の設定を見直すこと: ~A" e)))))
    (unless result
      (error "mutation testing を始める前に、既定のテストスイートが落ちている。~
まずテストを通してから mutation testing をかけること。"))))

(defun %print-report (report stream)
  (format stream "~&mutation testing 結果~%")
  (dolist (m (report-mutants report))
    (format stream "~&  [~A] ~A:~D~%    ~S~%    -> ~S~%"
            (mutant-status m) (namestring (mutant-file m)) (mutant-line m)
            (mutant-original-form m) (mutant-mutated-form m)))
  (let* ((mutants (report-mutants report))
         (total (length mutants))
         (excluded (count :excluded mutants :key #'mutant-status))
         (killed (count :killed mutants :key #'mutant-status))
         (timeout (count :timeout mutants :key #'mutant-status))
         (survived (count :survived mutants :key #'mutant-status)))
    (format stream "~&total=~D killed=~D timeout=~D survived=~D excluded=~D~%"
            total killed timeout survived excluded)
    (format stream "~&mutation score = ~A~%" (mutation-score report))))

(defun run (&key (system "nabla")
                 (test-system nil)
                 (test-function #'default-test-function)
                 files
                 ranges
                 (base-ref "main")
                 (exclusions (default-exclusions-path))
                 (timeout-seconds 300)
                 (trials 20)
                 (stream *standard-output*))
  "対象のファイル・行範囲（既定は BASE-REF から HEAD への git diff）に
含まれる定義に、演算子（*MUTATION-OPERATORS* の順）を1つずつ試し、
最初に適用できたものを1つの変異体として TEST-FUNCTION で判定する。
戻り値は REPORT。SYSTEM / TEST-SYSTEM は現時点では記録用で、
既定の TEST-FUNCTION の選択には使わない（DEFAULT-TEST-FUNCTION を見よ）。"
  (declare (ignore system test-system))
  (%check-baseline test-function)
  (let ((exclusion-list (%normalize-exclusions exclusions))
        (candidates (%collect-candidates (%ranges-for-run files ranges base-ref)))
        (mutants nil))
    (dolist (candidate candidates)
      (destructuring-bind (path . def) candidate
        (let ((original (source-form-form def)))
          (dolist (operator *mutation-operators*)
            (multiple-value-bind (mutated applied) (mutate-form original operator)
              (when applied
                (let* ((excluded-entry (excluded-p exclusion-list path operator original mutated))
                       (status (if excluded-entry
                                   :excluded
                                   (%status-for original mutated (source-form-package def)
                                                test-function timeout-seconds trials))))
                  (push (make-mutant :file path :line (source-form-start-line def)
                                      :original-form original :mutated-form mutated
                                      :operator operator :status status)
                        mutants))
                (return)))))))
    (let ((report (make-report :mutants (nreverse mutants))))
      (%print-report report stream)
      report)))

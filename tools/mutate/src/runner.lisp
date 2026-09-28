;;;; runner.lisp -- 変異体を作り、テストスイートを回して判定する

(in-package #:nabla.mutate)

(defstruct mutant
  "1つの変異体の記録。STATUS は :killed / :survived / :excluded / :timeout、
まだ判定していない（PLAN-MUTANTS や RUN の :DRY-RUN）なら NIL。
PACKAGE は定義を読んだときの *PACKAGE*（評価に使う。export しない）。"
  file line original-form mutated-form operator status package)

(setf (documentation 'mutant-file 'function) "その変異体があるファイルのパス。")
(setf (documentation 'mutant-line 'function) "その変異体の定義が始まる行番号（1始まり）。")
(setf (documentation 'mutant-original-form 'function) "変異させる前の、トップレベルの定義。")
(setf (documentation 'mutant-mutated-form 'function) "変異させた後の、トップレベルの定義。")
(setf (documentation 'mutant-operator 'function) "適用した変異演算子（*MUTATION-OPERATORS* のいずれか）。")
(setf (documentation 'mutant-status 'function)
      "判定結果。:killed / :survived / :excluded / :timeout のいずれか。未判定なら NIL。")

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
  ;; SB-SYS:INTERACTIVE-INTERRUPT（Ctrl-C）は SERIOUS-CONDITION のサブタイプ
  ;; だが、変異体のふるまいではなく利用者の中断なので :killed として
  ;; 記録せず、そのまま外へ signal し直して RUN を止める。
  (let ((*package* package))
    (unwind-protect
         (%isolate-mutant-side-effects
          (lambda ()
            (handler-case
                (progn
                  (%eval-quietly mutated-form)
                  (handler-case
                      (if (sb-ext:with-timeout timeout-seconds (funcall test-function))
                          :survived
                          :killed)
                    (sb-ext:timeout () :timeout)
                    (sb-sys:interactive-interrupt (c) (error c))
                    (serious-condition () :killed)))
              (sb-sys:interactive-interrupt (c) (error c))
              (serious-condition () :killed))))
      (ignore-errors (%eval-quietly original-form)))))

(defun %check-it-trials-symbol ()
  (let ((package (find-package "CHECK-IT")))
    (and package (find-symbol "*NUM-TRIALS*" package))))

(defparameter *regression-directories* (list "tests/regressions/")
  "check-it の :regression-file がファイルを書き込みうる、カレント
ディレクトリ相対のディレクトリのリスト。%ISOLATE-MUTANT-SIDE-EFFECTS が
各変異体の評価の前後でここの中身をまるごと退避・復元し、mutation
testing がリポジトリに副作用を残さないようにする。nabla では
tests/regressions/ がそれにあたる（tests/support/regression.lisp の
REGRESSION-PATH を見よ）。nabla-mutate は nabla のコアシステムに
依存しない独立したツールという設計なので、このパス自体は単なる既定値
であり、nabla 以外のプロジェクトで runner を使うときは RUN の
:REGRESSION-DIRECTORIES で上書きすること。")

(defun %directory-pathname (designator)
  "DESIGNATOR（文字列またはパス）を、末尾がディレクトリ区切りの
絶対パスにして返す。相対パスはカレントディレクトリからの相対とみなす
（run.sh がリポジトリ直下から実行する前提と合わせている）。"
  (let ((path (uiop:ensure-directory-pathname designator)))
    (if (uiop:absolute-pathname-p path)
        path
        (merge-pathnames path (uiop:getcwd)))))

(defun %walk-regular-files (dir)
  "DIR（ディレクトリの絶対パス）以下の通常ファイルを再帰的にすべて
集めて返す。DIR が存在しなければ NIL。"
  (when (uiop:directory-exists-p dir)
    (append (uiop:directory-files dir)
            (mapcan #'%walk-regular-files (uiop:subdirectories dir)))))

(defun %snapshot-directory (dir)
  "DIR 以下の全ファイルの内容を、DIR からの相対パスをキーにした
alist として返す。ファイルはテキストとして読む（regression ファイルは
Lisp のソースなので、この前提で問題ない）。"
  (let ((base (%directory-pathname dir)))
    (mapcar (lambda (file)
               (cons (enough-namestring file base)
                     (alexandria:read-file-into-string file)))
             (%walk-regular-files base))))

(defun %restore-directory (dir snapshot)
  "DIR の中身を SNAPSHOT（%SNAPSHOT-DIRECTORY が返した alist）の状態へ
戻す。SNAPSHOT になかった今あるファイル（変異体が新しく作ったもの）は
消し、SNAPSHOT にあった内容は（変更されていても消されていても）
書き戻す。"
  (let* ((base (%directory-pathname dir))
         (kept (make-hash-table :test #'equal)))
    (dolist (entry snapshot)
      (setf (gethash (car entry) kept) t))
    (dolist (file (%walk-regular-files base))
      (let ((relative (enough-namestring file base)))
        (unless (gethash relative kept)
          (ignore-errors (delete-file file)))))
    (dolist (entry snapshot)
      (let ((path (merge-pathnames (car entry) base)))
        (ensure-directories-exist path)
        (with-open-file (stream path :direction :output
                                      :if-exists :supersede
                                      :if-does-not-exist :create)
          (write-string (cdr entry) stream))))))

(defun %snapshot-directories (dirs)
  (mapcar (lambda (dir) (cons dir (%snapshot-directory dir))) dirs))

(defun %restore-directories (snapshots)
  (dolist (entry snapshots)
    (%restore-directory (car entry) (cdr entry))))

(defun %check-it-regression-indicator ()
  "check-it が regression-case を積む plist の indicator シンボル
（CHECK-IT::REGRESSION-CASES）を返す。check-it がロードされていなければ
NIL。"
  (let ((package (find-package "CHECK-IT")))
    (and package (find-symbol "REGRESSION-CASES" package))))

(defun %snapshot-regression-plists (indicator)
  "INDICATOR（check-it::regression-cases）を今持っているすべての
シンボルについて、(シンボル . 値) の alist を返す。DO-ALL-SYMBOLS で
image 全体を1回舐める。INDICATOR が NIL（check-it 未ロード）なら NIL。"
  (let ((snapshot nil))
    (when indicator
      (do-all-symbols (sym)
        (let ((value (get sym indicator '%not-present)))
          (unless (eq value '%not-present)
            (push (cons sym value) snapshot)))))
    snapshot))

(defun %restore-regression-plists (indicator snapshot)
  "今 INDICATOR を持っているすべてのシンボルを、SNAPSHOT の状態へ戻す。
SNAPSHOT になければ REMPROP し、あれば元の値に戻す（SNAPSHOT にあって
今は消えている、まず起きないはずのケースも念のため戻す）。"
  (when indicator
    (let ((restored (make-hash-table :test #'eq)))
      (do-all-symbols (sym)
        (let ((value (get sym indicator '%not-present)))
          (unless (eq value '%not-present)
            (let ((entry (assoc sym snapshot)))
              (if entry
                  (setf (get sym indicator) (cdr entry))
                  (remprop sym indicator)))
            (setf (gethash sym restored) t))))
      (dolist (entry snapshot)
        (unless (gethash (car entry) restored)
          (setf (get (car entry) indicator) (cdr entry)))))))

(defun %isolate-mutant-side-effects (thunk)
  "THUNK を呼ぶ間に *REGRESSION-DIRECTORIES* に書き込まれたファイルと、
check-it の regression-cases（シンボルの plist）への変更を、呼び終わった
あと必ず元に戻す。1つの変異体（または baseline チェック）の評価が、
ディスクにも image にも副作用を残して次の評価に混ざらないようにする
汎用の仕組み（tools/mutate/README.md の「regression 状態の隔離」を
見よ）。THUNK の戻り値をそのまま返す。"
  (let ((dir-snapshot (%snapshot-directories *regression-directories*))
        (indicator (%check-it-regression-indicator)))
    (let ((plist-snapshot (%snapshot-regression-plists indicator)))
      (unwind-protect (funcall thunk)
        (%restore-directories dir-snapshot)
        (%restore-regression-plists indicator plist-snapshot)))))

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
  (let ((result (%isolate-mutant-side-effects
                 (lambda ()
                   (handler-case (funcall test-function)
                     (error (e)
                       (error "mutation testing を始める前に、既定のテストスイートが~
エラーで終わった。:test-function か :test-system の設定を見直すこと: ~A" e)))))))
    (unless result
      (error "mutation testing を始める前に、既定のテストスイートが落ちている。~
まずテストを通してから mutation testing をかけること。"))))

(defun %minimal-difference (original mutated)
  "ORIGINAL と MUTATED の違いを含む、いちばん内側の部分フォームの組を
(values before after) で返す。同じ長さのリストで違う要素がちょうど
1つで、その要素どうしがどちらもリストなら、その中へ降りる（atom の
違いは、それを含むフォームごと示したほうが読みやすいので降りない）。"
  (let ((length-a (and (consp original) (ignore-errors (list-length original))))
        (length-b (and (consp mutated) (ignore-errors (list-length mutated)))))
    (if (and length-a (eql length-a length-b))
        (let ((diffs (loop for a in original
                           for b in mutated
                           for index from 0
                           unless (equal a b) collect index)))
          (if (and (= 1 (length diffs))
                   (consp (nth (first diffs) original))
                   (consp (nth (first diffs) mutated)))
              (%minimal-difference (nth (first diffs) original) (nth (first diffs) mutated))
              (values original mutated)))
        (values original mutated))))

(defun %print-mutant (m stream)
  "変異体1つを、変わった部分だけの短い形で印字する。生き残った変異体は、
除外リストの :mutation にそのまま写せるよう、定義全体の組も印字する。"
  (multiple-value-bind (before after)
      (%minimal-difference (mutant-original-form m) (mutant-mutated-form m))
    (let ((*print-pretty* nil) (*print-length* 8) (*print-level* 4)
          (*package* (or (mutant-package m) *package*)))
      (format stream "~&  [~A] ~A:~D ~(~A~)~%"
              (or (mutant-status m) "PLANNED") (namestring (mutant-file m)) (mutant-line m)
              (mutant-operator m))
      (if (eq (mutant-operator m) :delete-form)
          ;; 消したフォームが *PRINT-LENGTH* より後ろにあると、前後の
          ;; フォームが同じに印字されてしまうので、消したものだけを示す。
          (format stream "    削除: ~S~%"
                  (nth (or (mismatch before after :test #'equal) 0) before))
          (format stream "    ~S -> ~S~%" before after))))
  (when (eq (mutant-status m) :survived)
    (format stream "    :mutation ~S~%"
            (%mutation-string (mutant-original-form m) (mutant-mutated-form m))))
  (finish-output stream))

(defun %print-summary (report stream)
  (let* ((mutants (report-mutants report))
         (total (length mutants))
         (excluded (count :excluded mutants :key #'mutant-status))
         (killed (count :killed mutants :key #'mutant-status))
         (timeout (count :timeout mutants :key #'mutant-status))
         (survived (count :survived mutants :key #'mutant-status)))
    (format stream "~&total=~D killed=~D timeout=~D survived=~D excluded=~D~%"
            total killed timeout survived excluded)
    (when (zerop total)
      ;; total=0 は「対象範囲に変異させられる定義が1つもなかった」ことの
      ;; 証拠であって、良いスコアの証拠ではない。git diff の範囲抽出が
      ;; （設定やパスの不一致で）何も拾えなかった場合もこの形になるので、
      ;; mutation score = 1 だけを見て安心しないよう、はっきり警告する。
      (format stream "~&警告: 変異させられる定義が1つも見つからなかった。~
対象範囲（--base / FILE[:START-END] や git diff の設定）を確認すること。~
mutation score = 1 はこの場合「良い結果」ではない。~%"))
    (format stream "~&mutation score = ~A~%" (mutation-score report))))

(defun %sample-evenly (list cap)
  "LIST から CAP 個を、先頭から末尾まで等間隔に選ぶ（決定的）。
CAP が NIL か LIST の長さ以下なら LIST をそのまま返す。"
  (let ((length (length list)))
    (if (or (null cap) (<= length cap))
        list
        (loop for i below cap
              collect (nth (floor (* i length) cap) list)))))

(defun %mutants-for-definition (path def cap)
  "定義 DEF の変異体を、*MUTATION-OPERATORS* の順・演算子ごとに前順で
すべて作る。変異後のフォームが同じもの（:constant と :off-by-one が
同じ値を選んだときなど）は1つにまとめ、CAP があれば等間隔に間引く。"
  (let ((original (source-form-form def))
        (seen (make-hash-table :test #'equal))
        (mutants nil))
    (dolist (operator *mutation-operators*)
      (dolist (mutated (mutation-sites original operator))
        (unless (gethash mutated seen)
          (setf (gethash mutated seen) t)
          (push (make-mutant :file path :line (source-form-start-line def)
                             :original-form original :mutated-form mutated
                             :operator operator :package (source-form-package def))
                mutants))))
    (%sample-evenly (nreverse mutants) cap)))

(defun plan-mutants (&key files ranges (base-ref "main") max-mutants-per-definition)
  "テストを走らせずに、RUN が作る変異体（STATUS は NIL）のリストを
返す。FILES / RANGES / BASE-REF は RUN と同じ意味。
MAX-MUTANTS-PER-DEFINITION が正の整数なら、1つの定義あたりの変異体を
その数まで等間隔に間引く（NIL なら間引かない）。同じ入力には常に
同じ順の同じ変異体を返す。"
  (loop for (path . def) in (%collect-candidates (%ranges-for-run files ranges base-ref))
        nconc (%mutants-for-definition path def max-mutants-per-definition)))

(defun run (&key (system "nabla")
                 (test-system nil)
                 (test-function #'default-test-function)
                 files
                 ranges
                 (base-ref "main")
                 (exclusions (default-exclusions-path))
                 (timeout-seconds 300)
                 (trials 20)
                 max-mutants-per-definition
                 dry-run
                 (regression-directories *regression-directories*)
                 (stream *standard-output*))
  "対象のファイル・行範囲（既定は BASE-REF から HEAD までの差分で
変わった行）に含まれる定義について、*MUTATION-OPERATORS* の各演算子を
適用できるすべての箇所に1つずつ変異体を作り（PLAN-MUTANTS）、
TEST-FUNCTION で判定する。戻り値は REPORT。
MAX-MUTANTS-PER-DEFINITION は PLAN-MUTANTS と同じ。DRY-RUN が真なら
baseline もテストも走らせず、作る予定の変異体（STATUS は NIL）を
印字してそのまま REPORT にして返す（変異体の数だけを安く数えたいとき）。
SYSTEM / TEST-SYSTEM は現時点では記録用で、既定の TEST-FUNCTION の
選択には使わない（DEFAULT-TEST-FUNCTION を見よ）。
REGRESSION-DIRECTORIES は check-it の :regression-file が書き込みうる
ディレクトリのリスト（既定 *REGRESSION-DIRECTORIES*）。各変異体（と
baseline チェック）の評価の前後で、この中身と check-it の
regression-cases plist をまるごと退避・復元し、評価どうしで副作用が
混ざらないようにする。"
  (declare (ignore system test-system))
  (let ((*regression-directories* regression-directories)
        (plan (plan-mutants :files files :ranges ranges :base-ref base-ref
                            :max-mutants-per-definition max-mutants-per-definition)))
    (if dry-run
        (%print-plan plan stream)
        (%run-mutation-loop plan test-function exclusions timeout-seconds trials stream))))

(defun %print-plan (plan stream)
  (dolist (m plan)
    (%print-mutant m stream))
  (format stream "~&mutated-definitions=~D mutants=~D~%"
          (length (remove-duplicates (mapcar (lambda (m)
                                               (cons (namestring (mutant-file m)) (mutant-line m)))
                                             plan)
                                     :test #'equal))
          (length plan))
  (make-report :mutants plan))

(defun %run-mutation-loop (plan test-function exclusions timeout-seconds trials stream)
  (%check-baseline test-function)
  ;; 結果は1体ずつ、判定したそばから印字する。変異体が SBCL ごと落とす
  ;; （pseudo-atomic 中の制御スタック枯渇など、Lisp から捕まえられない
  ;; 致命的エラー）と最後の一覧は出ないので、どこまで進んだか
  ;; （次の変異体が落とした犯人であること）をこれで分かるようにする。
  (format stream "~&mutation testing 結果（~D 体）~%" (length plan))
  (let ((exclusion-list (%normalize-exclusions exclusions)))
    (dolist (m plan)
      (setf (mutant-status m)
            (if (excluded-p exclusion-list (mutant-file m) (mutant-operator m)
                            (mutant-original-form m) (mutant-mutated-form m))
                :excluded
                (%status-for (mutant-original-form m) (mutant-mutated-form m)
                             (mutant-package m) test-function timeout-seconds trials)))
      (%print-mutant m stream))
    (let ((report (make-report :mutants plan)))
      (%print-summary report stream)
      report)))

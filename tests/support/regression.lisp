;;;; regression-path: tests/regressions/ の下のファイルを管理する。

(in-package #:nabla.tests.support)

(defun regression-path (name &key (package "NABLA.TESTS"))
  "tests/regressions/NAME.lisp のパスを返す。

check-it は :regression-file に渡したファイルが既に存在することを
要求する (:if-does-not-exist :error) ので、ここでファイルが無ければ
作る。1行目には (in-package ...) を書く。これは check-it が保存する
regression-case フォームを、そのパッケージで読み込めるようにするため。

PACKAGE (文字列またはシンボル) が in-package 先になり、既定は
呼び出し元のテスト本体が通常属している NABLA.TESTS。かつては呼び出し
時の *package* をそのまま使っていたが、scripts/run-tests.sh のように
トップレベルの *package* が COMMON-LISP-USER のまま check-it がこの
関数を呼ぶ経路では、意図しない (in-package #:common-lisp-user) が
書き込まれてしまっていた。REPL や mutation runner など NABLA.TESTS
以外のパッケージで regression ファイルを読み込ませたいときは、この
引数で明示的に指定する。"
  (let ((path (asdf:system-relative-pathname
               "nabla" (format nil "tests/regressions/~A.lisp" name))))
    (ensure-directories-exist path)
    (unless (probe-file path)
      (with-open-file (stream path :direction :output :if-does-not-exist :create)
        (format stream "~&(in-package #:~(~A~))~%" (string package))))
    path))

(defun load-package-regressions (package)
  "tests/regressions/ のうち、1 行目が (in-package #:PACKAGE) のファイルをすべて load する。

nabla/tests の tests/regressions.lisp は全ファイルを load するが、nabla/tests は他のテスト
システムに依存しないので、その時点で無いパッケージのファイルは読み飛ばす。そのため、各テスト
システムは自分のパッケージを作った直後にこれを呼び、自分の回帰ケースを登録する（呼ばないと、
check-it が保存した失敗例が再生されない）。PACKAGE は文字列かシンボル。"
  (let ((header (format nil "(in-package #:~(~A~))" (string package))))
    (dolist (path (sort (uiop:directory-files
                         (asdf:system-relative-pathname "nabla" "tests/regressions/")
                         "*.lisp")
                        #'string<
                        :key #'namestring))
      (when (string= header (with-open-file (stream path) (read-line stream nil "")))
        (load path)))))

;;; check-it のバグへの対処（issue #220）。check-it.lisp の SAVE-REGRESSION は、失敗例を
;;; ファイルに追記するのと同時に (push string (get regression-id 'regression-cases)) で
;;; 生の文字列をそのまま積む。ところが再生側（check-it%）は積まれたものを REGRESSION-CASE
;;; オブジェクトとみなして (datum regression-case) を呼ぶ。そのため、同じプロセスで失敗した
;;; あとに同じ id でもう一度 check-it を走らせる（同じテストの再実行、id の共有）と、
;;; NO-APPLICABLE-METHOD CHECK-IT::DATUM ("94169") で落ち、本当の失敗が隠れる。
;;; ファイルから load した REGRESSION-CASE は、DATUM を read-from-string した値を持つので、
;;; 文字列に対する DATUM も同じ読み方で値を返す。
;;;
;;; 残す理由（issue #245）: 回帰 id をテストごとに分けること（#220 の (1)）と
;;; regressions/ids-are-unique（(2)）が防ぐのは、別のテストが同じ id で再生する経路 (a) だけ。
;;; 同じテストが失敗した同じプロセスで、もう一度実行される経路 (b)（REPL や
;;; fiveam:run! の再実行）と、1つのテストの中で同じ :regression-id の check-it を dtype ごとに
;;; ループで呼ぶ経路 (c)（tests/iree/reduce-test.lisp、tests/vmap-elementwise-test.lisp）は防げない。
;;; (c) は新しいプロセスで 1 回走らせる scripts/run-tests.sh / CI でも、失敗が出れば踏む。
;;; docs/check-it-repros/string-datum.lisp で、(a)(b)(c) のどれもこのメソッドが無いと
;;; NO-APPLICABLE-METHOD になることを確かめた。
;;; tools/mutate の runner は変異体ごとに regression-cases の plist を元に戻すので、
;;; runner 自体は (b) を踏まない（tools/mutate/src/runner.lisp の %isolate-mutant-side-effects）。
;;; 上流（check-it の固定コミットは master の HEAD で、dev などの他ブランチにも修正は無い）の issue は未報告（ユーザーの確認待ち）。
;;; 下書きは docs/check-it-upstream-bugs.md。報告したらここにリンクを書く。
;;; check-it::datum は check-it の内部シンボルなので、上流が直ったら外す
;;; （条件は .claude/skills/nabla-testing/SKILL.md の「check-it の固定コミットを上げる条件」）。
(defmethod check-it::datum ((datum string))
  (read-from-string datum))

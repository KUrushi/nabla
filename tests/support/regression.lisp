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

;;;; regression-path: tests/regressions/ の下のファイルを管理する。

(in-package #:nabla.tests.support)

(defun regression-path (name)
  "tests/regressions/NAME.lisp のパスを返す。

check-it は :regression-file に渡したファイルが既に存在することを
要求する (:if-does-not-exist :error) ので、ここでファイルが無ければ
作る。1行目には (in-package ...) を、呼び出し時の *package* を使って書く。
これは check-it が保存する regression-case フォームを、そのパッケージで
読み込めるようにするため。

呼び出し時の *package* をそのまま使うので、FiveAM のテスト本体（通常
NABLA.TESTS）から呼ぶことを前提にしている。REPL や mutation runner など
別のパッケージ（例えば COMMON-LISP-USER）から呼ぶと、1行目の in-package
がそのパッケージになる点に注意。check-it が保存するシンボルはパッケー
ジ修飾つきで印字されるので実害はないが、初回に作られるファイルの
見た目は呼び出し元の *package* に依存する。"
  (let ((path (asdf:system-relative-pathname
               "nabla" (format nil "tests/regressions/~A.lisp" name))))
    (ensure-directories-exist path)
    (unless (probe-file path)
      (with-open-file (stream path :direction :output :if-does-not-exist :create)
        (format stream "~&(in-package #:~(~A~))~%" (package-name *package*))))
    path))

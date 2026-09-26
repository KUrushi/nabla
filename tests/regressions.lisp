;;;; tests/regressions/ の下にある回帰テストのファイルを、すべて load する。
;;;;
;;;; check-it が失敗例を保存するときは、そのファイルに
;;;; (check-it:regression-case ...) フォームを追記する。ここで load して
;;;; おかないと、次回以降そのケースが再実行されない。

(in-package #:nabla.tests)

(dolist (path (sort (directory
                      (asdf:system-relative-pathname "nabla" "tests/regressions/*.lisp"))
                     #'string<
                     :key #'namestring))
  (load path))

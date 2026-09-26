;;;; tests/regressions/ の下にある回帰テストのファイルを、すべて load する。
;;;;
;;;; check-it が失敗例を保存するときは、そのファイルに
;;;; (check-it:regression-case ...) フォームを追記する。ここで load して
;;;; おかないと、次回以降そのケースが再実行されない。

(in-package #:nabla.tests)

;;; ASDF の system-relative-pathname は "*.lisp" をワイルドカードとしてでは
;;; なく、リテラルなファイル名の一部として解釈する（parse-unix-namestring の
;;; 挙動）。そのため (directory ...) にそのまま渡しても常に NIL になり、
;;; regressions/ 以下のファイルが1つも load されない。uiop:directory-files
;;; はディレクトリと globパターンを別引数で受け取るので、これを使う。
(dolist (path (sort (uiop:directory-files
                      (asdf:system-relative-pathname "nabla" "tests/regressions/")
                      "*.lisp")
                     #'string<
                     :key #'namestring))
  (load path))

(in-suite :nabla.small)

(test regressions/loader-registers-committed-cases
  "tests/regressions/*.lisp が実際に load され、check-it の
regression-case が登録されていることを確かめる。ここが NIL に戻ったら、
regressions.lisp のグロブが再び効かなくなった（今回の直った不具合の再発）
ということ。"
  (is (>= (length (get 'support/allclose/boundary 'check-it::regression-cases))
          1)))

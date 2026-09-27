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
;;;
;;; tests/regressions/ は nabla/tests 以外のテストシステム（nabla/iree/tests
;;; など）も regression-path 経由で共有しており、そこに書かれるファイルの
;;; 1行目は (in-package #:nabla.iree.tests) のような、そのシステム自身の
;;; パッケージを指す。nabla/tests は nabla/iree/tests に依存しない
;;; （システム構成は契約どおり独立）ので、nabla/tests を単体でロードする
;;; ときはそのパッケージがまだ存在せず、load がエラーになる。そのシステムの
;;; 側で（load-time に）自分の regression ファイルを読み直すので、ここでは
;;; パッケージが無いことによる失敗は警告に落として読み飛ばす。
(dolist (path (sort (uiop:directory-files
                      (asdf:system-relative-pathname "nabla" "tests/regressions/")
                      "*.lisp")
                     #'string<
                     :key #'namestring))
  (handler-case (load path)
    (sb-ext:package-does-not-exist ()
      (warn "tests/regressions.lisp: ~A のパッケージが今はロードされていないので読み飛ばした（そのテストシステム自身がロード時に読み直す）" path))))

(in-suite :nabla.small)

(test regressions/loader-registers-committed-cases
  "tests/regressions/*.lisp が実際に load され、check-it の
regression-case が登録されていることを確かめる。ここが NIL に戻ったら、
regressions.lisp のグロブが再び効かなくなった（今回の直った不具合の再発）
ということ。"
  (is (>= (length (get 'support/allclose/boundary 'check-it::regression-cases))
          1)))
